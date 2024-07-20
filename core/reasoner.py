import openai 
import json
import os

class Reasoner:
    def __init__(self, gpt_model_version):
        self.api_key = ""
        openai.api_key = self.api_key
        self.gpt_model_version = gpt_model_version
        with open('../data/dataset/manuals/{self.database}_manuals_data.json', 'r') as f:
            self.manuals_data = json.load(f)
        with open('../data/dataset/historical_questions/{self.dataset}_retrieval_data.json', 'r') as f:
            self.historical_questions_data = json.load(f)
        self.all_retrieval_data = {}
        for key, value in self.manuals_data.items():
            self.all_retrieval_data[rf'm_{key}'] = value
        for key, value in self.historical_questions_data.items():
            self.all_retrieval_data[rf'question_{key}'] = value


    def generate_prompt(self, query, retrieved_docs):
        
        retrival_str = ''
        q_count = 0
        m_count = 0
        q_text = ''
        m_text = ''
        for index, d_id in enumerate(retrieved_docs):
            d_id = d_id['question_id']
            if "question_" in d_id:
                question_id = d_id.replace("question_", "")
                knobs = self.historical_questions_data[question_id]['parameter']
                cand_parameters = cand_parameters | set(knobs)
                q_text += rf"{q_count+1}. Question: {self.historical_questions_data[question_id]['question']}. Parameters: {knobs}; "
                q_count += 1
            else:
                manual_id = d_id.replace("manual_", "")
                knobs = self.manuals_data[manual_id]['parameters']
                cand_parameters = cand_parameters | set(knobs)
                m_text += rf"{m_count+1}. Manual: {self.manuals_data[manual_id]['text']}. Parameters: {knobs}; "
                m_count += 1
        cand_parameters = list(cand_parameters)


        demo = "Below are some user questions and the corresponding recommended parameters: "
        demo += q_text + '.'
        
        demo += "Here are some documentations that you can refer: "
        demo += m_text + '.'

        demo += rf"Below are some corresponding recommended candidate parameters from above information: {str(cand_parameters)}"

        prompt = rf"Assume you are a DBA and you are particularly skilled at recommending database parameters. {demo}. Please answer the following query and recommend parameters: {query}. Please recommend parameters occured in reference questions and manuals. Please output list as ['parameter1', 'parameter2', ...]."
        return prompt

    def apply(self, prompt):
        self.gpt_client = openai.ChatCompletion.create(
            model=self.gpt_model_version,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return self.gpt_client.choices[0].message['content']