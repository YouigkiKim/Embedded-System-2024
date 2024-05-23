import torch
import torch.onnx
import onnx

# 모델 정의 (예시로 간단한 모델 사용)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

# 모델 인스턴스 생성 및 초기화
model = SimpleModel()

# 모델 파라미터 로드 (여기서는 학습된 .pth 파일 경로 사용)
model.load_state_dict(torch.load('intersection_right.pth'))

# 모델을 평가 모드로 설정
model.eval()

# 더미 입력 생성 (모델 입력에 맞는 크기로)
dummy_input = torch.randn(1, 10)

# ONNX 파일로 모델 변환
torch.onnx.export(model,               # 실행할 모델
                  dummy_input,         # 모델 입력 (튜플 또는 여러 입력들도 가능)
                  "intersection_right.onnx",        # 저장될 모델의 이름
                  export_params=True,  # 모델 파일 내 파라미터 저장
                  opset_version=10,    # ONNX 버전 설정
                  do_constant_folding=True,  # 최적화: 상수 폴딩 사용
                  input_names = ['image'],   # 모델 입력 이름
                  output_names = ['x'], # 모델 출력 이름
                )

# ONNX 모델 검증
onnx_model = onnx.load("intersection_right.onnx")
onnx.checker.check_model(onnx_model)
