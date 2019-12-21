# How to upload pypi?

## How to

- [graphtorch] 폴더안에 __init__.py 를 생성
  - 빈 파일이도 상관없음, 폴더를 패키지로 인식하게 해주는 역할
- 현재 graphtorch 폴더안에느 아래와 같은 .py 파일이 존재함
  - matrix.py
  - model.py
- matrix.py안의 SparseMatrix를 불러오려면?
```python
from grpahtorch.matrix import SparseMatrix
```

- 아래와 같이 **[graphtorch] 폴더 밖에(같은 디렉토리에)** setup.py 파일을 생성
```python
from setuptools import setup, find_packages


setup(
    name             = 'graphtorch',
    version          = '0.1',
    description      = 'Package converts sparse graph to matrix',
    long_description = open('README.md').read(),
    author           = 'Hyeonwoo Yoo',
    author_email     = 'hyeon95y@gmail.com',
    url              = 'https://github.com/KU-BIG/graphtorch',
    download_url     = 'https://github.com/KU-BIG/graphtorch',
    packages         = find_packages(),
    classifiers      = [
        'Programming Language :: Python :: 3.6'
    ]
)
```

- 아래와 같이 **[graphtorch] 폴더 밖에(같은 디렉토리에)** setup.cfg 파일을 생성
```
[metadata]
description-file = README.md
```

- 아래와 같이 **[graphtorch] 폴더 밖에(같은 디렉토리에)** MENIFEST.in 파일을 생성
```
include LICENSE
include README.md
```

- 아래와 같이 **[graphtorch] 폴더 밖에(같은 디렉토리에)** README.md 파일을 생성
```
필요 내용 작성
```


- 업로드에 필요한 패키지 설치
```
pip3 install wheel
```

- referneces를 따라하는 와중에 아래 패키지 에러가 나서 추가로 업데이트해주었음
```
pip3 install --upgrade keyrings.alt
```

- wheel로 배포 파일 만들기 
```
python3 setup.py sdist bdist_wheel
```
- twine으로 패키지 업로드
```
python3 -m twine upload dist/*
```

## References

- [파이썬 코딩도장 : 45.3 패키지 만들기](https://dojang.io/mod/page/view.php?id=2449)
- [TroubleShooting : Keyring error](https://stackoverflow.com/questions/53164278/missing-dependencies-causing-keyring-error-when-opening-spyder3-on-ubuntu18)
- [파이썬 모듈 pypi에 배포하는 방법](https://devlog.jwgo.kr/2018/03/11/how-to-deploy-to-pypi/)
- [PyPI로 패키지 배포하기 : 내가 만드 모듈로 pip로 다운받을 수 있다!](https://blessingdev.wordpress.com/2019/05/31/pypi로-패키지-배포하기내가-만든-모듈도-pip로-다운받을/)
- [How to upload your python package to PyPi](https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56)
