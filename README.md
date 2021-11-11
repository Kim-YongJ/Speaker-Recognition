# Speaker-Recognition(화자 인식)
독립적인 인물들의 음성데이터를 사전에 학습하여 말하는 사람이 누구인지 파악 할 수 있다.   
## 화자 인식의 종류
### 화자 검증(Speaker Verification)
<p align="center">
  <img src="https://user-images.githubusercontent.com/19161231/47408492-946e6b80-d79a-11e8-8ae1-8d118cbd65da.png">
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/19161231/47408073-d0083600-d798-11e8-8dc0-23bb62664bb4.png">
</p>
<p align="center">< 그림 1 : 화자 검증 개요도 > </p>   
`화자 검증`은 저장한 화자의 음성과 입력 음성 사이의 유사도(Likelihood)를 이용하여 구한다.   
즉, 저장되지 않은 목소리, 즉, 사칭 목소리와  ‘유사도 간 비율(Δ)’을 측정한다.   
그리고 사칭자 대비 신뢰할 수 있는 비율에 해당하는 ‘기준값(Θ)’에 따라 화자 일치 여부를 검증하게 된다.   
그러한 과정으로 시스템은 이를 통해 등록되지 않은 목소리를 구별할 수가 있게 된다.
  
### 화자 식별(Speaker Identification)
<p align="center">
  <img src="https://user-images.githubusercontent.com/19161231/47408080-d3032680-d798-11e8-92ce-c3a28712fe18.png">
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/19161231/47408093-e0201580-d798-11e8-9fff-e671183c5c5a.png">
</p>
<p align="center">< 그림 2 : 화자 식별 개요도 ></p>   
`화자식별`은 미리 N명의 화자 모델(Voiceprint)을 등록한 시스템을 이용하여 화자 식별을 한다(그림2).   
특정 사람이 시스템에 음성을 입력하면, 등록된 화자 모델을 검색해 음성과 가장 일치하는 화자를 찾게 된다.   
현재 프로젝트에서는 화자 식별을 이용한 화자인식을 Convolutional Neural Network를 이용하여 구현하였다.      
## MFCC
<p align="center">  
  <img src="https://user-images.githubusercontent.com/19161231/47408767-984ebd80-d79b-11e8-9665-f6d895348234.png" width="60%">
</p>
<p align="center">< 그림 3 : MFCC ></p>   
`MFCC`는 Mel Frequency Cepstrum Coefficient의 약자로서 음성인식 분야에서 널리 사용되는 알고리즘이다.   
MFCC는 소리의 특징을 추출하는 알고리즘으로서 입력받은 소리를 일반적으로 20ms-40ms정도의 작은 프레임으로 쪼개는 과정을 거치고 이러한 프레임들의 스펙트럼을 분석하여 특징을 추출하는 기법이다.   
MFCC를 이용한 Features 추출은 음정이 변해도 어느 정도 일정하다는 장점이 있기 때문에 음성인식에 효과적인 알고리즘이다.    
아래의 그림은 MFCC의 전체적인  과정이다.   
<p align="center">  
  <img src="https://user-images.githubusercontent.com/19161231/47408769-98e75400-d79b-11e8-9599-93cc7bdb9b67.png" width="70%">
</p>
<p align="center">< 그림 4 : MFCC 동작 과정 ></p>  
사람마다 개개인의 고유한 DNA 및 지문을 가지고 있듯이, 각 사람마다 목소리의 고유 주파수, 진폭 등이 다르다는 특성(MFCC)을 이용하여 개인의 신분을 확인 할 수 있는 화자 인식이 가능하다.   
현재 프로젝트에서는 음성 전처리 과정을 MFCC를 사용해서 진행하고 있다.       
## Project Structure
<p align="center">
  <img src="https://user-images.githubusercontent.com/19161231/47409509-ec5aa180-d79d-11e8-9edf-c1c2de89a80a.png">
</p>
<p align="center">< 그림 5 : Project Structure ></p>  
## Data Collection
<p>
  <table>
    <tr>
      <th> Folder name </th>
      <th> Name </th>
      <th> Number of data </th>
    </tr>
    <tr>
      <td align="center">00</td>
      <td align="center">유인나</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">01</td>
      <td align="center">배철수</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">02</td>
      <td align="center">이재은</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">03</td>
      <td align="center">최일구</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">04</td>
      <td align="center">문재인</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">05</td>
      <td align="center">김용준</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">06</td>
      <td align="center">이재빈</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">07</td>
      <td align="center">강은기</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">08</td>
      <td align="center">이명박</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">09</td>
      <td align="center">노무현</td>
      <td align="center">20</td>
    </tr>
    <tr>
      <td align="center">10</td>
      <td align="center">박근혜</td>
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
    <td align="center"> 학습방법 1 </td>
    <td align="center"> 학습방법 2 </td>
  </tr>
  <tr>
    <td align="center"><img src="https://user-images.githubusercontent.com/55169156/121474787-8af3a980-c9ff-11eb-97ae-7e2827a8df07.png"></td>
    <td align="center"><img src="https://user-images.githubusercontent.com/55169156/121474751-78797000-c9ff-11eb-93b7-642b8fad0013.png"></td>    
  </tr>
</table>
</p>
학습방법 1은 사전에 5개의 class에 대해서 학습을 진행한 다음 새로운 데이터를 포함해서 총 11개의 class에 대해서 재학습을 진행하는 방법이고   
학습방법 2는 새로운 데이터를 포함해서 총 11개의 class에 대해서 학습을 진행하는 방법이다.   
## Result
<p>
<table>  
  <tr>
    <th colspan="2">
      Result
    </th>
  </tr>  
  <tr>
    <td align="center"> 학습방법 1 </td>
  </tr>
  <tr>
    <td align="center"><img src="https://user-images.githubusercontent.com/55169156/121475300-30a71880-ca00-11eb-980c-68c595b7c58f.png"></td>    
  </tr>
  <tr>
    <td align="center"> 학습방법 2 </td>
  </tr>
  <tr>
    <td align="center"><img src="https://user-images.githubusercontent.com/55169156/121475341-3dc40780-ca00-11eb-805c-20486fb2675b.png"></td>    
  </tr>
</table>
</p>
<p>
전체적으로 각 class 별 정확도가 큰 차이가 나지 않지만 학습방법 1같은 경우    
학습방법 2보다 더 적은 학습 횟수와 시간을 가지기 때문에 추후에 사용하게 될 훈련 방법은    
사전에 미리 학습을 시킨 다음에 새로운 데이터가 들어올 경우 다시 재학습하는 방법을 사용하게 될 것이다.    
</p>
