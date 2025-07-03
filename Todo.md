# TODO

- roi 사각형을 쓰레드로 구현 - 큐를 만들어서 frame 전달. 큐 사이즈 = 1 최신 frame 만 작업 할 수 있게. - 항상 roi를 그리기, yolo가 변수를 변경. - 변수를 인식해서 roi 항상 그리기.
  -> 속도 체크.

- 송출을 ffmpeg 로 process. stdin.write 로 송출
  -> rtmp://localhost/live/stream 접속해서 이미지 확인 - localtunnel 로 ip 받아서 접속! - localtunnel 로 stream 을 wan으로 링크 -> 코드 수정
