import openai 
import json
import os

class Reasoner:
    def __init__(self,dataset, data_path, gpt_model_version):
        self.api_key = ""
        self.data_path = data_path
        openai.api_key = self.api_key
        self.gpt_model_version = gpt_model_version
        self.dataset = dataset
        self.database = dataset.split("_")[0]
        with open(rf'{self.data_path}/dataset/manuals/{self.database}_manuals_data.json', 'r') as f:
            self.manuals_data = json.load(f)
        with open(rf'{self.data_path}/dataset/historical_questions/{self.dataset}_retrieval_data.json', 'r') as f:
            self.historical_questions_data = json.load(f)


    def generate_prompt(self, query, retrieved_docs):
        
        retrival_str = ''
        q_count = 0
        m_count = 0
        q_text = ''
        m_text = ''
        cand_parameters = set()
        for index, d_index in enumerate(retrieved_docs):
            if d_index >= len(self.manuals_data):
                question_id = list(self.historical_questions_data.keys())[d_index-len(self.historical_manuals_data)]
                knobs = self.historical_questions_data[question_id]['parameter']
                cand_parameters = cand_parameters | set(knobs)
                q_text += rf"{q_count+1}. Question: {self.historical_questions_data[question_id]['question']}. Parameters: {knobs}; "
                q_count += 1
            else:
                manual_id = list(self.manuals_data.keys())[d_index]
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