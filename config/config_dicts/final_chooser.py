from config.model import FullPayload
final_config = {
    "struct": 
    [{
        "prompt": """

        Context:

        {context}
        ---

        You are an expert data scientist and handler tasked with recommending actions to a user
        Your instructions: 
            - The context provides model specs and their evaluation metrics, from only a part of the data. Recommend the most promising three models to the user to train on the full data

            - The model specs will be in the slug format: scaling_method-model-imbalance_handling_method, in this order

            - You will provide in the actions dict two keys: True, and False. True will contain the list of the three recommended models. They will be in the same slug format given: scaling-model-imbalance, copy the names exactly, and put the rest in the False dict. If there are less than three models, just assign whatever is there based on your recommendation

            - you will also provide reasoning for your actions in the reasoning key

            - You will also create a prompt for the user (in the prompts attribute (just one string item in the list)) showing them your recommendations, and explaining why you chose them and ask the user to choose three models to fully train.

            - in the beginning, create a ### heading with the task name, then put your prompt

            - Use markdown for good stylization and professional language

        """,
        "schema": FullPayload,
        "output_name": "fin0",
        "context_demands": {'update': 'model_metrics'}
    }],
    "interrupt_name": ["final"],
    }
