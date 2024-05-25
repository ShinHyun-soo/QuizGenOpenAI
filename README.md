### (free/ online database server 찾기)
- mongodb json format 저장, 조금 사용하면 무료
- 마리아 db, 리눅스, 조금 사용하면 무료
- 
### 추가 사항
- 채점 결과를 문제에 바로 표시할 수 있게
### (2024-05-17)회의록 요약
https://github.com/MrBounty/streamlit-google-auth
#### 변경 사항 및 요구사항
1. **언어 지원**
2. 한글 먼저 나오게 ✅
   - 스페인어, 프랑스어 삭제 ✅

3. **질문 형식**
   - Open-end 질문을 단답형, 서술형으로 변경

4. **영상 및 캡션**
   - 유튜브 영상 시청 가능하게 ✅
   - 모든 영상에 캡션 추가 ✅
   - Context 제공 필요 (텍스트는 제외) ✅

5. **질문 퀄리티 평가 프롬프트**
   - 사용자 계정 생성: 스트림릿(Streamlit)으로 가능 여부 확인 ✅

#### 기능 구현
1. **문제 출력**
   - JSON, e-class 출제자 포맷, pydantic 파싱, HTML 포맷으로 내보내기
   - 다운로드 버튼 추가
   - eclass 포맷으로 출력 가능하게

2. **답 저장 위치**
   - 현재 객체 프로퍼티로 저장

3. **문제 난이도** 
   - Easy, Normal, Hard로 구현 및 프롬프트에 추가 ✅

4. **청크 선택**
   - 긴 컨텍스트에서 임의로 청크 선택
   - 유튜브 문제 생성 시 큰 텍스트는 청크로 분할, 청크 표시 및 선택 가능하게
   - 청크 선택 버튼은 최대 10개 이하

5. **사용자 LLM 선택** ✅
   - GPT-3.5 문제점, GPT-4.0 사용 필요
   - 무료 버전 Ollama 지원

6. **프롬프트 활용**
   - 단답형 문제: 프롬프트로 가능
   - 빈칸 채우기 문제: 프롬프트로 pydantic 출력 형식 설정 어려움

7. **추가 요구사항**
   - 로컬 서버 구축 시도
   - 결과 카톡에 공유 
   - 대외용 사이트 글자 모두 한글로 작성 ✅
   - 이전 생성 문제 제거 및 새로 고침 기능 추가
   - 출력 형식에서 문제 번호 제거 ✅
   - 한국어 기준으로 문제 의미 파악 

#### 향후 일정
- 다음 주 수요일 11시 상상파크+에서 회의

#### 작업 분담 및 정리 필요
- 각 작업 항목과 담당자 정리 필요

#### 그 외
- UI 통일화 하였음. ✅
- 채점 결과에서 정답을 숨길 수 있게 만들었음. ✅

#### 버그
- 유튜브 라이브 링크 입력시 오류 발생. ✅ (단순 오류 였음.)
- 객관식에서 답이 거의 1번.
- 리붓시 db 가 날아감
- 로그인 정보 페이지간 연동
- 중복 계정 생성

#### 피드백
- 라디오 버튼이 Unchecked 되게 ✅
- 사용자가 직접 문제 요구사항을 입력할 수 있게 ✅
- 문제 답이 안 맞는 경우가 있음.

# Runs(Local)
```python
streamlit run SubToQuiz.py
```

# Runs(Site)
[quiz-bot-4](https://quiz-bot-4.streamlit.app/)
