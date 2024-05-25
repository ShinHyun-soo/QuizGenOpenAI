from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from typing import List

class QuizMultipleChoice(BaseModel):
    quiz_text: str = Field(description="The quiz test")
    questions: List[str] = Field(description="The quiz quesitons")
    alternatives: List[List[str]] = Field(description="The quiz alternatives")
    answers: List[str] = Field(description="The quiz answers")

class QuizTrueFalse(BaseModel):
    quiz_text: str = Field(description="The quiz test")
    questions: List[str] = Field(description="The quiz quesitons")
    alternatives: List[List[str]] = Field(description="The quiz alternatives")
    answers: List[str] = Field(description="The quiz answers")

class QuizOpenEnded(BaseModel):
    questions: List[str] = Field(description="The quiz quesitons")
    answers: List[str] = Field(description="The quiz answers")


def create_quiz_chain(prompt_template, llm, pydantic_object_schema):
    """Creates the chain for the quiz app."""
    return prompt_template | llm.with_structured_output(pydantic_object_schema)


def create_multiple_choice_template(language):
    """Create the prompt template for the quiz app, including conditional translation."""
    template = """ 
    You are an expert quiz maker for technical fields. Let's think step by step and
    create a {difficulty} quiz with {num_questions} multiple-choice questions about the following concept/content: {quiz_context}.

    {user_input}

    The format of the quiz should be as follows:

    - Multiple-choice: 
    - Questions:
        <Question1>: 
            - Alternatives1: <option 1>, <option 2>, <option 3>, <option 4>
        <Question2>: 
            - Alternatives2: <option 1>, <option 2>, <option 3>, <option 4>
        ....
        <QuestionN>: 
            - AlternativesN: <option 1>, <option 2>, <option 3>, <option 4>
    - Answers:
        <Answer1>: <option 1 | option 2 | option 3 | option 4>
        <Answer2>: <option 1 | option 2 | option 3 | option 4>
        ....
        <AnswerN>: <option 1 | option 2 | option 3 | option 4>
    """

    # Conditionally add translation instruction based on the selected language
    if language != "English":
        template += f"\n\nPlease ensure that the quiz is accurately translated into {language}, maintaining the technical accuracy and clarity of the questions and options."


    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def create_true_false_template(language):
    """Create the prompt template for the quiz app."""

    template = """
    You are an expert quiz maker for technical fields. Let's think step by step and
    create a {difficulty} quiz with {num_questions} questions about the following concept/content: {quiz_context}.
    
    {user_input}

    The format of the quiz could be one of the following:
    - True-false:

    - Questions:
        <Question1>: 
            - Alternatives1: <True>, <False>
        <Question2>: 
            - Alternatives2: <True>, <False>
        .....
        <QuestionN>: 
            - AlternativesN: <True>, <False>
    - Answers:
        <Answer1>: <True|False>
        <Answer2>: <True|False>
        .....
        <AnswerN>: <True|False>
    """
    # Conditionally add translation instruction based on the selected language
    if language != "English":
        template += f"\n\nPlease ensure that the quiz is accurately translated into {language}, maintaining the technical accuracy and clarity of the questions and options."


    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def create_open_ended_template(language):
    template = """
    You are an expert quiz maker for technical fields. Let's think step by step and
    create a quiz with {num_questions} questions about the following concept/content: {quiz_context}.
    
    {user_input}

    The format of the quiz could be one of the following:
    - Open-ended:
    - Questions:
        <Question1>: 
        <Question2>:
        .....
        <QuestionN>:
    - Answers:    
        <Answer1>:
        <Answer2>:
        .....
        <AnswerN>:

    """
    # Conditionally add translation instruction based on the selected language
    if language != "English":
        template += f"\n\nPlease ensure that the quiz is accurately translated into {language}, maintaining the technical accuracy and clarity of the questions and options."


    prompt = ChatPromptTemplate.from_template(template)
    return prompt