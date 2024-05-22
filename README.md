***Embedded-System-2024

File 
---
prac22.py 
* 주행 중 촬영을 위한 코드
* selec버튼을 사용해 레코딩 On/off

***task1.py
 - 자율주행 구현코드
 - 표지판 인지모델 결합 전의 모델

***task2.py
 - task1과 task2가 합쳐진 최종모델
 - 조이스틱 6(L1),7(R1)번 키를 사용해 주행 중 throttle조정 기능 추가
 - 타임플래그를 활용한 표지판 제어 구현
   1. 횡단보도
       - 최초 인식 시 flag활성화 및 최초 인식 시간체크 ->  현재시간 체크 5초가 지나면 예외처리를 통해 인식 중에도 주행가능하도록 구현
   3. 버스
       - 인식 시점부터 저속주행 시작, 표지판 사라진 이후 2초간 서행 유지
   5. 교차로
       - 방향별로 차선 인식모델 학습
       - 표지판 인지 시 direction변수 방향에 알맞게  변경
       - 제어문을 통해 direction변수와 알맞은 모델로 이미지 입력 후 x값 출력
      
       
