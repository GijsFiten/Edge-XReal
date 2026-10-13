"""Validate the 4070 environment, build LDIF, and smoke-test the three models.
Use: bash run_4070.sh setup_4070.py --build --smoke
No package installation or environment mutation is performed.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGES=('Total3DUnderstanding','Implicit3DUnderstanding','ThreeDSceneFormer')
MODELS=('Total3D','Implicit3D','3DSceneFormer')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if sys.platform!='linux' or Path(sys.prefix).name!='hsu':
        raise SystemExit('Use conda hsu inside Ubuntu')
    root=Path(__file__).resolve().parent
    prefix=Path(sys.prefix)
    if os.environ.get('CUDA_HOME')!=str(prefix):
        raise SystemExit('Run via bash run_4070.sh setup_4070.py')
    out=root/'revision_results'/(datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%SZ')+'-4070-setup')
    out.mkdir(parents=True)
    print('Logs: '+str(out),flush=True)
    def run(name, command, timeout=300):
        print(name+' ...',flush=True)
        with (out/(name+'.log')).open('w') as log:
            log.write(json.dumps(command)+'\n'); log.flush()
            import signal
            p=subprocess.Popen(command,cwd=str(root),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL); p.wait()
                raise RuntimeError(name+' timed out; see '+str(out))
        if code: raise RuntimeError(name+' failed; see '+str(out))
    # No model imports until a GPU allocation, matrix multiply, and detector op pass.
    run('gpu', [sys.executable,'-u','-c', '''import torch, torchvision
assert torch.__version__.split('+')[0]=='2.0.1', torch.__version__
assert torchvision.__version__.split('+')[0]=='0.15.2', torchvision.__version__
assert torch.version.cuda=='11.8', torch.version.cuda
assert torch.cuda.get_device_capability()==(8,9), torch.cuda.get_device_capability()
print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda,flush=True)
x=torch.ones((32,32),device='cuda'); assert (x@x).sum().item()==32768
b=torch.tensor([[0.,0.,1.,1.]],device='cuda')
assert torchvision.ops.nms(b,torch.ones(1,device='cuda'),.5).numel()==1
torch.cuda.synchronize(); print('GPU and torchvision NMS passed',flush=True)
'''])
    # Assets are transferred separately; never substitute weights or regenerate priors.
    manifest=json.loads((root/'inference_assets.json').read_text())
    missing=[]
    for item in manifest['files']:
        p=root/item['path']
        if not p.is_file() or sha(p)!=item['sha256']: missing.append(item['path'])
    if missing: raise RuntimeError('Missing or mismatched assets: '+', '.join(missing))
    for package in PACKAGES:
        for name in ('layout_avg_file.pkl','size_avg_category.pkl'):
            source=root/name
            dest=root/package/'data/sunrgbd/preprocessed'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and sha(dest)!=sha(source):
                raise RuntimeError('Refusing to overwrite different prior: '+str(dest))
            if not dest.exists(): shutil.copyfile(source,dest)
    for model in MODELS:
        run('preflight-'+model,[sys.executable,str(root/'run_revision_tests.py'),'--suite','preflight','--model',model,'--limit','5','--out',str(out/('preflight-'+model))])
    if args.build:
        nvcc=prefix/'bin/nvcc'
        compiler=prefix/'bin/x86_64-conda-linux-gnu-g++'
        if not nvcc.is_file() or not compiler.is_file():
            raise RuntimeError('Conda CUDA toolkit or C++ compiler missing')
        version=subprocess.check_output([str(nvcc),'--version'],universal_newlines=True)
        if 'release 11.8' not in version: raise RuntimeError('Expected Conda nvcc 11.8: '+version)
        (out/'nvcc.txt').write_text(version)
        for package in PACKAGES[1:]:
            mesh=root/package/'external/ldif/ldif2mesh'
            binary=mesh/'ldif2mesh'
            temporary=mesh/('ldif2mesh-'+out.name)
            if binary.exists(): shutil.copyfile(binary,out/(package+'-previous-ldif2mesh'))
            run('build-'+package,[str(nvcc),'-ccbin',str(compiler),'-Xptxas','-O3','-gencode=arch=compute_89,code=sm_89','-gencode=arch=compute_89,code=compute_89','--ptxas-options=-v','-maxrregcount','63','--std','c++11',str(mesh/'ldif2mesh.cu'),'-o',str(temporary)],timeout=600)
            grid=out/(package+'-test.grd')
            run('kernel-'+package,[str(temporary),str(mesh/'test-ldif.txt'),str(mesh/'extracted.occnet'),str(grid),'-resolution','32','-extent','0.75'])
            if not grid.exists() or grid.stat().st_size==0: raise RuntimeError('LDIF did not generate a grid')
            temporary.replace(binary)
    if args.smoke:
        for model in MODELS:
            run('smoke-'+model,[sys.executable,str(root/'run_revision_tests.py'),'--suite','smoke','--model',model,'--out',str(out/('smoke-'+model))],timeout=1800)
    print('Requested checks passed. Review logs; full benchmarks have not run.',flush=True)

if __name__=='__main__': main()
