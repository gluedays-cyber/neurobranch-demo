# NeuroBranch Demo Manual & Guide

<p align="center">
  <img src="assets/neurobranch.jpg" width="100%" alt="NeuroBranch">
</p>

## 1. 개요 (Overview)

본 문서는 **NeuroBranch Intelligent Branching Demo**의 실행 및 아키텍처 매뉴얼이다.
순수 Go 언어로 빌드된 초경량 내장 신경망 라우터 엔진을 활용하여, 정적 `if/else` 및 정규식의 취약성과 클라우드 LLM의 네트워크 지연 및 비용 문제를 해결하는 지능형 분기(Intelligent Branching) 실습 코드를 다룬다.

---

## 2. 프로젝트 디렉토리 구조

```text
neurobranch-demo/
├── assets/
│   └── neurobranch.jpg        # 타이틀바 배너 이미지
├── data/
│   └── train.csv              # 도메인 학습 데이터셋 (Query, Intent)
├── weights/
│   └── model.bin              # 컴파일된 가중치 바이너리 (v3 포맷)
├── go.mod                     # Go 모듈 정의
├── main.go                    # 데모 실행 엔트리포인트 (5대 패턴 시연)
├── main_test.go               # 전수 기능 검증 단위 테스트
├── README.md                  # 프로젝트 소개 문서
└── MANUAL.md                  # 데모 상세 가이드 및 매뉴얼
```

---

## 3. 핵심 7대 지능형 분기 패턴 (v3.0 규격)

| 패턴 | 함수/형식 | 특성 및 목적 |
| :--- | :--- | :--- |
| **Functional Options** | `TrainAIWithOptions(samples, cfg, opts...)` | 신뢰도 임계치, LogSumExp 에너지 컷오프 등 옵션 외부화 주입 |
| **Native Switch** | `switch branch := ai.Select(query); branch` | 자연어 변형을 Go 표준 switch 레이블로 직접 매핑 (`default:` 폴백) |
| **Guard Clause** | `if ai.If(query, "Refund") { ... }` / `ai.Is` | 단일 의도 타겟팅 및 신뢰도 기반 조기 탈출 분기 |
| **Comma-ok Pattern** | `if intent, ok := ai.Match(query); ok { ... }` | 결정적(true) vs 모호한(false) 의도 판별 및 OOD 안전 격리 |
| **Declarative DSL** | `ai.Switch(query).Case(...).Auto(...).Confirm(...).Default(...).Evaluate(ctx)` | 자동 실행(`Auto`), 사용자 확인(`Confirm`), 기본 예외(`Default`) 플루언트 체이닝 |
| **Dynamic Tuning** | `ai.SetEnergyThreshold(...)` / `ai.SetTemperature(...)` | 무재학습 동적 에너지 가드 임계치 및 Softmax 온도 스케일링 튜닝 |
| **Atomic Hot-Swap** | `ai.AppendDataMap(...)` / `ai.Reload(path)` | 무중단 무잠금(Lock-Free) 0 ns 가중치 교체 및 신규 도메인 실시간 주입 |

---

## 4. 실행 및 테스트 방법

### 4.1 전체 단위 테스트 실행

```bash
go test -v ./...
```

### 4.2 데모 애플리케이션 실행

```bash
go run main.go
```

