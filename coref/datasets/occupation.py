from datasets.common import names

jobs = [
    ("stocks shelves", " store"),
    ("delivers mail", " post"),
    ("arrests criminals", " police"),
]

template = """Answer the question based on the context below. Keep the answer short. Respond "Unsure about answer" if not sure about the answer.

Context: {subject_1} {action_1}. {subject_2} {action_2}.

Question: Where does {qn_subject} work?

Answer: {qn_subject} works at the"""


def generate_prompt(name1, job1, name2, job2, qn_name):
    prompt = template.format(
        subject_1=name1,
        action_1=job1[0],
        subject_2=name2,
        action_2=job2[0],
        qn_subject=qn_name,
    )
    assert qn_name == name1 or qn_name == name2
    if qn_name == name1:
        answer = (job1[1], job2[1])
    else:
        answer = (job2[1], job1[1])
    return prompt, answer
