# Speaker-Recognition(화자 인식)
 사람의 음성의 특징을 추출하여 인식하는 화자 인식(Speaker Recognition) 기술은 특정 문장의 사용에서의 인식에 따라 문장 종속(Text dependent)방식과 문장 독립(Text Independent)방식으로 구분된다. 문장 종속방식은 한 사람 목소리의 고유한 개별 특성을 학습함으로써 작동하고, 화자 등록 과정과 검증 과정의 음성의 내용이 동일하기 때문에 높은 정확도를 얻을 수 있다. 반대로 문장 독립방식은 불특정 다수 화자의 음성을 인식하도록 개발되고, 검증 과정에서의 음성의 내용이 등록한 음성의 내용과 무관하기에 상대적으로 높은 정확도를 얻기 힘들다. 
    
## Data Preprocessing
<p align="center">  
  <img src="MFCC_1.jpg" width="70%">
</p>
<p align="center">< 그림 1 : MFCC ></p>
  
### STFT
<p>
 STFT는 시간, 주파수와 음량 간의 상관관계를 손실하지 않고 특징을 추출 해내기 위해서 시간 구간별로 Fourier Transform 연산을 수행하는 알고리즘이다.
 본 논문에서 적용한 수치는 샘플링 레이트, Hop Length, n_mels, n_fft이고 샘플링 레이트는 이산적 신호를 얻기 위한 단위 시간당 샘플링 횟수를 뜻하고, Hop Length는 음성의 magnitude를 얼마나 겹처서 잘라서 보여주는 지를 뜻하고, n_mels는 주파수를 나눠주는 수이고, n_fft는 음성 프레임의 길이를 결정하는 것이다. 각 수치는 16,000, 256, 40, 512이다.
 </p>
 <p align="center">
  <img src="STFT.jpg" width="60%">
 </p>
 <p align="center">< 그림 2 : STFT  ></p>
 
### Windowing
 <p>
  Window 함수는 시간 연속적인 신호의 특정 구간을 취하여 분석하기 위해 사용되며, 다양한 종류가 있으나 목적과 신호특성에 따라 최적의 Window 함수를 선택하는 것이 중요하다. 본 논문에서는 일반적으로 사용하는 Hamming 함수와 Hanning 함수중 Hanning 함수를 사용하여 신호의 불연속성을 제거하는 작업을 거쳤다. 
 </p>
  <p align="center">
  <img src="WindowFuction.jpg" width="60%">
 </p>
 <p align="center">< 그림 3 : Windowing  ></p>
 
### MFCC
<p>
 `MFCC`는 Mel Frequency Cepstrum Coefficient의 약자로서 음성인식 분야에서 널리 사용되는 알고리즘이다.   
MFCC는 소리의 특징을 추출하는 알고리즘으로서 입력받은 소리를 일반적으로 20ms-40ms정도의 작은 프레임으로 쪼개는 과정을 거치고 이러한 프레임들의 스펙트럼을 분석하여 특징을 추출하는 기법이다.   
MFCC를 이용한 Features 추출은 음정이 변해도 어느 정도 일정하다는 장점이 있기 때문에 음성인식에 효과적인 알고리즘이다.    
아래의 그림은 MFCC의 전체적인  과정이다.
  </p>
<p align="center">  
  <img src="MFCC.jpg" width="60%">
</p>
<p align="center">< 그림 4 : MFCC 동작 과정 ></p>  
 사람마다 개개인의 고유한 DNA 및 지문을 가지고 있듯이, 각 사람마다 목소리의 고유 주파수, 진폭 등이 다르다는 특성(MFCC)을 이용하여 개인의 신분을 확인 할 수 있는 화자 인식이 가능하다.   
본 프로젝트에서는 음성 전처리 과정을 MFCC를 사용해서 진행하였다.
 </p>
 
 ### Liftering
 <p>
 주변 소음을 제거하여 높은 인식률을 보이는 MFCC를 잡음이 있는 환경에서 성능을 높이기 위해 De-Emphasize 하는 Lifter 함수를 적용하였다. Lifter 연산은 전체 Cepstrum에서 원하는 위치의 직사각형 Window를 곱해 분석할 주파수 영역을 선택하는 주파수 영역에서의 필터링 연산이다.
 </p>
 
## CNN Model Structure
<p align="center">
  <img src="Cnn_Model_Structure.jpg">
</p>
<p align="center">< 그림 5 : CNN Model Structure ></p> 
 
## Data Collection
 <p>
   본 논문의 실험에 사용한 음성 데이터 셋은 11명의 화자에게 문장 독립방식 화자 인식을 위한 통일된 15개의 문장과 사용자가 임의로 정한 5개의 문장, 총 사용자별로 20개씩 220개의 5초의 음성 데이터로 구성되어 있다. 이 음성 데이터들은 44.1khz로 샘플링된 후 wav 파일로 변환 후 3장에서 제안된 전처리 과정을 통해 npy 파일로 변환 후 임의로 학습 데이터 80%와 테스트데이터 20%로 비율을 설정하여 제안된 Conv1D 모델을 통해 화자별 음성 데이터의 학습을 진행한 후 화자별로 테스트를 진행하였다.
  </p>
<p>
  <table>
    <tr>
      <th> Folder name </th>
      <th> Name </th>
      <th> Number of data </th>
    </tr>
    <tr>
      <td align="center">00</td>
      <td align="center">강은기</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">01</td>
      <td align="center">강은서</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">02</td>
      <td align="center">김나리</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">03</td>
      <td align="center">김엄지</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">04</td>
      <td align="center">김용준</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">05</td>
      <td align="center">윤동준</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">06</td>
      <td align="center">이성민</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">07</td>
      <td align="center">이재빈</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">08</td>
      <td align="center">정승화</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">09</td>
      <td align="center">주다영</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">10</td>
      <td align="center">주홍식</td>
      <td align="center">20</td>
    </tr>
  </table>
</p>

## Library
```python
import pyaudio                     # Record audio from mic
import wave                        # Read/Write .wav
import scipy                       # Use signal
import os                          # Access folder
import librosa                     # Calc mfcc
import librosa.display             # Display mfcc with matplotlib
import matplotlib.pyplot as plt    # Draw waveform
import tensorflow as tf            # Use tf
import keras                       # Use keras
import numpy as np
import pandas as pd
```

## Learning process
<p>
<table>  
  <tr>
    <th colspan="2">
      Learning process
    </th>
  </tr>  
  <tr>
    <td align="center"> 학습방법</td>
  </tr>
  <tr>
    <td align="center"><img src="Learing.jpg"></td>  
  </tr>
</table>
</p>
학습방법은 새로운 데이터를 포함해서 총 11개의 class에 대해서 학습을 진행하는 방법이다.

## Result
<p>
<table>  
  <tr>
    <th colspan="2">
      Result
    </th>
  </tr>  
  <tr>
    <td align="center"> 사용자별 화자인식 정확도 </td>
  </tr>
  <tr>
    <td align="center"><img src="Result.jpg"></td>    
  </tr>
</table>
</p>
<p>
 11명의 사람의 1 ~ 19번 음성 데이터를 학습한 후 20번 음성 데이터로 모델의 성능 테스트를 한 결과를 표로 정리하여 나타낸 모습이다. 16 ~ 20번 음성 데이터는 사용자가 임의로 정한 같은 문장을 반복함으로 해당 문장의 인식 정확도를 테스트하기 위한 목적으로 설정되었다. 학습과 테스트를 5번 반복하여 사용자별로 평균 인식 정확도를 확인하였다.
 </p>
 <p>
 위의 결과를 통해 모델의 사용자별 화자 인식 정확도는 높게는 99~100%, 낮게는 75%가 나옴을 확인할 수 있다. 남자 사용자의 경우 평균 정확도가 88%의 성능을 보이고 높게는 93%까지의 성능을 보여준다. 여자 사용자의 경우는 사용자별 차이가 있지만, 평균 정확도가 92%의 성능을 보여준다.  
</p>
