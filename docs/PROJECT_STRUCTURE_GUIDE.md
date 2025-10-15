# OpenRouter MCP 프로젝트 구조 가이드

## 📁 디렉토리 구조 원칙

### 루트 디렉토리 (/): 필수 파일만
- `README.md`: 프로젝트 메인 문서
- `CONTRIBUTING.md`: 기여 가이드  
- `LICENSE`: 라이선스
- `claude_mcp.py`: 메인 CLI 인터페이스
- `package.json`, `requirements*.txt`: 의존성 관리
- `*.ini`: 설정 파일들
- `.env.example`: 환경 변수 템플릿 (실제 .env는 제외)

### 소스코드 (/src/): 모든 Python 코드
- `openrouter_mcp/`: 메인 패키지
  - `client/`: API 클라이언트 코드
  - `handlers/`: MCP 핸들러들
  - `collective_intelligence/`: 집단 지성 시스템
    - The purpose of this project is to provide an external tool that will allow running multiple heavy models in parralel (like 8 gpt5-pro instances + gemini 2.5 pro + grok 4) and then helps to ensemble reductive reasoning so help models re-run with each other's cross-arguments and help deduce the best answer based on that. Each new attempt should run less agents. If there is a high level of agreement - run much less models on the second run. More disagreement - more parralel runs next time. Up to 4 possible total runs. The last step is using the most powerful model to assemble the most advanced answer possible combining the power of the most powerful competetive models from OpenAI, Google, Anthropic and xAI (and more)
    - 목적: 여러 고성능 모델을 병렬로 실행하고 반복적·감소형 앙상블 추론으로 최종 고품질 응답을 생성하는 외부 오케스트레이터 도구
  - `models/`: 데이터 모델 및 캐시
  - `config/`: 설정 관리
  - `utils/`: 유틸리티 함수들

### 테스트 (/tests/): 모든 테스트 코드
- **원칙**: 루트에 `test_*.py` 파일 금지
- `test_client/`: 클라이언트 테스트
- `test_handlers/`: 핸들러 테스트  
- `test_collective_intelligence/`: 집단 지성 테스트

### 문서 (/docs/): 모든 문서
- **가이드 문서**: 사용법, 설치, 설정 관련
- `reports/`: 벤치마크 보고서, 테스트 결과 등

### 벤치마크 (/benchmarks/): 성능 측정
- **벤치마크 스크립트**: 성능 측정 코드
- **결과 파일**: 자동 생성되는 결과들 (.gitignore로 제외)

### 스크립트 (/scripts/): 유틸리티 스크립트
- 설치 스크립트, 개발 도구 등

## 🚫 금지 사항

### 루트에 두면 안 되는 파일들
- `test_*.py`: tests/ 디렉토리로
- `*_test.py`: tests/ 디렉토리로  
- `*_results_*.json`: .gitignore로 제외
- `*_report_*.json/md`: docs/reports/로
- `debug_*.py`: 개발 후 삭제
- `quick_*.py`: 개발 후 삭제
- 하드코딩된 API 키가 있는 파일

### 보안 위험 파일들
- API 키가 하드코딩된 파일
- 실제 환경 변수가 포함된 .env 파일
- 개인 정보가 포함된 테스트 파일

## ✅ 파일 배치 가이드

### 새로운 테스트 파일
```
# 잘못된 위치
./test_new_feature.py  ❌

# 올바른 위치  
./tests/test_new_feature.py  ✅
```

### 벤치마크 관련
```
# 스크립트
./benchmarks/new_benchmark.py  ✅

# 결과 (자동 .gitignore)
./benchmarks/results_*.json  ✅
```

### 문서
```
# 보고서
./docs/reports/feature_analysis.md  ✅

# 가이드
./docs/FEATURE_GUIDE.md  ✅
```

## 🔄 정기 정리 체크리스트

### 주간 정리 (매주)
- [ ] 루트 디렉토리에 test_*.py 파일 없는지 확인
- [ ] 임시 결과 파일들 (.gitignore 패턴) 정리
- [ ] docs/reports/에 보고서 파일들 정리

### 월간 정리 (매월)
- [ ] 벤치마크 결과 파일 아카이브
- [ ] 중복 테스트 파일 통합
- [ ] 문서 중복성 검토

### 릴리스 전 정리
- [ ] API 키 하드코딩 전수 검사
- [ ] 불필요한 개발/디버그 파일 제거
- [ ] .gitignore 패턴 검증
- [ ] 프로젝트 구조 이 가이드 준수 확인

## 🛠️ 자동화 도구

### Git Hooks (권장)
```bash
# pre-commit: API 키 하드코딩 검사
git config core.hooksPath .githooks
```

### 정리 스크립트
```bash
# 개발 완료 후 정리
python scripts/cleanup_dev_files.py

# 벤치마크 결과 아카이브  
python scripts/archive_benchmarks.py
```

---

**마지막 업데이트**: 2025-08-13  
**버전**: v1.0