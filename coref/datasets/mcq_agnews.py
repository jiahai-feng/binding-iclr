import copy
from coref.datasets.overthinking_datasets import (
    get_dataset,
    AGNEWS_DATASET_PARAMS,
    AGNEWS_LABELS,
)
from coref.datasets.mcq import MCQDataset, MCQ, MCQVocab
from coref.datasets.common import BaseVocab, TrackFormatter, Substring, MCQStatement


class Vocab(MCQVocab):
    type = "MCQ_AGNEWS"

    def __init__(self, tokenizer_type):
        super().__init__(tokenizer_type, AGNewsDataset(), ["A", "B", "C", "D"])


class AGNewsDataset(MCQDataset):
    @classmethod
    def num_options(cls):
        assert 4 == len(AGNEWS_LABELS[0])
        return len(AGNEWS_LABELS[0])

    def get_qn(self, answer, key):
        """
        Returns an MCQ instance
        """
        return self.dataset[answer][key]

    def get_length(self, answer):
        return len(self.dataset[answer])

    def __init__(self):
        dataset = get_dataset(AGNEWS_DATASET_PARAMS)
        assert len(dataset) == len(AGNEWS_LABELS[0])
        # Flatten and process
        self.dataset = [
            [
                MCQ(
                    question=question[0].replace("\\", " "),
                    options=AGNEWS_LABELS[0],
                    answer=answer,
                )
                for question in subset
            ]
            for answer, subset in enumerate(dataset)
        ]


class TEMPLATES:
    @classmethod
    def lookup(cls, template_type):
        if template_type == "normal":
            return cls.DEFAULT
        if template_type == "reversed":
            return cls.REVERSED
        if template_type == "reversedkshot":
            return cls.REVERSED_KSHOT
        raise Exception(f"unknown template {template_type}")

    class DEFAULT:
        @classmethod
        def lookup(cls, context_type):
            if context_type == "normal":
                return cls.context_template
            raise Exception(f"unknown context template {context_type}")

        @classmethod
        def lookup_qn(cls, vocab, qn_rng, opt_perm, query_name, raw_query_name):
            assert raw_query_name is None, "Raw query name not supported"
            real_query = opt_perm[query_name]
            return vocab.dataset.get_qn(
                real_query, qn_rng.integers(vocab.dataset.get_length(real_query))
            )

        @classmethod
        def generate_template(cls, *, vocab, template_context, prompt_id, context):
            @lambda f: f(**template_context)
            def ret(query_name, raw_query_name, **kwargs):
                assert prompt_id is not None
                qn_rng, opt_perm, label_perm = vocab.get_shuffled_labels(prompt_id)
                # pick first qn in the permutation
                cur_qn = cls.lookup_qn(
                    vocab=vocab,
                    qn_rng=qn_rng,
                    opt_perm=opt_perm,
                    query_name=query_name,
                    raw_query_name=raw_query_name,
                )
                return cls.template, {"question": cur_qn.question}

            return ret

        @classmethod
        def instantiate(cls, vocab, statement, prompt_id, template_context):
            @lambda f: f(**template_context)
            def ret(query_name, raw_query_name, **kwargs):
                context_template = cls.lookup(statement.type)
                qn_rng, opt_perm, label_perm = vocab.get_shuffled_labels(prompt_id)
                qn = cls.lookup_qn(
                    vocab=vocab,
                    qn_rng=qn_rng,
                    opt_perm=opt_perm,
                    query_name=query_name,
                    raw_query_name=raw_query_name,
                )
                option = qn.options[opt_perm[statement.option]]
                label, answer_label = vocab.labels[label_perm[statement.label]]
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    context_template,
                    label=label,
                    option=option,
                )
                return (
                    cur_ctx,
                    {
                        "label": ctx_idx_map["label"][0],
                        "option": ctx_idx_map["option"][0],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )

            return ret

        @classmethod
        def extract_template_indices(cls, full_output_indices):
            return {}

        template = """Article: {question}
Question: What is the best classification for this article?{context}
Answer:"""  # this follows https://openreview.net/pdf?id=yKbprarjc5B
        context_template = """\n{label}. {option}"""

    class REVERSED(DEFAULT):
        template = """Answer the question using the following options:{context}
Article: {question}
Question: What is the best classification for this article?
Answer:"""

    class REVERSED_KSHOT(DEFAULT):
        template = """{examplars}

Answer the question using the following options:{context}
Article: {question}
Question: What is the best classification for this article?
Answer:"""
        examplar_template = """\n\nAnswer the question using the following options:{context}
Article: {question}
Question: What is the best classification for this article?
Answer: {answer}"""

        @classmethod
        def generate_template(cls, *, vocab, template_context, prompt_id, context):
            @lambda f: f(**template_context)
            def ret(
                query_name, raw_query_name, n_examplars=1, same_context=False, **kwargs
            ):
                assert prompt_id is not None
                qn_rng, opt_perm, label_perm = vocab.get_shuffled_labels(prompt_id)
                # pick first qn in the permutation
                cur_qn = cls.lookup_qn(
                    vocab=vocab,
                    qn_rng=qn_rng,
                    opt_perm=opt_perm,
                    query_name=query_name,
                    raw_query_name=raw_query_name,
                )
                # print(f'true answer: {cur_qn.options[cur_qn.answer]}')
                new_qn_rng, context_rng = qn_rng.spawn(2)
                examplar_list = []
                for _ in range(n_examplars):
                    pseudo_option = new_qn_rng.integers(vocab.dataset.num_options())
                    examplar_qn = cls.lookup_qn(
                        vocab=vocab,
                        qn_rng=qn_rng,
                        opt_perm=opt_perm,
                        query_name=pseudo_option,
                        raw_query_name=None,
                    )
                    statement_texts = []
                    if not same_context:
                        pseudo_labels = context_rng.permutation(4)
                        pseudo_options = context_rng.permutation(4)
                        context = [
                            MCQStatement(label, option, "normal")
                            for label, option in zip(pseudo_labels, pseudo_options)
                        ]
                    real_option_label_map = {
                        opt_perm[statement.option]: label_perm[statement.label]
                        for statement in context
                    }
                    for statement in context:
                        label, answer_label = vocab.labels[label_perm[statement.label]]
                        option = examplar_qn.options[opt_perm[statement.option]]
                        cur_ctx, ctx_idx_map = TrackFormatter().format(
                            cls.context_template,
                            label=label,
                            option=option,
                        )
                        statement_texts.append(cur_ctx)

                    examplar_list.append(
                        cls.examplar_template.format(
                            question=examplar_qn.question,
                            answer=vocab.labels[
                                real_option_label_map[examplar_qn.answer]
                            ][1],
                            context="".join(statement_texts),
                        )
                    )
                    # print(f'true examplar answer: {examplar_qn.options[examplar_qn.answer]}')
                return cls.template, {
                    "question": cur_qn.question,
                    "examplars": "".join(examplar_list),
                }

            return ret
