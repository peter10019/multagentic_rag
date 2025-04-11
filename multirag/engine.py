from transformers import PreTrainedModel, PreTrainedTokenizerFast, StoppingCriteriaList

from multirag.utils import StopOnStringMatch


class ModelEngine:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerFast,
                 summarize_past: bool = True,
                 **generate_kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.summarize_past = summarize_past
        self.generate_kwargs = generate_kwargs

    def get_stopping_criteria(self, stop_sequences: list[str]) -> StoppingCriteriaList:
        return StoppingCriteriaList([StopOnStringMatch(stop_strings=stop_sequences, tokenizer=self.tokenizer)])

    def __call__(self, messages: [list[dict[str, str]]], stop_sequences: list[str] = '<end_action>') -> str:
        # chat templates for the model don't support tool-response role,changing it to user
        if self.summarize_past and len(messages) > 4:
            # Get past observations except past errors
            past_observations = [m['content'] for m in messages[:-1]
                                 if ((m['role'] == 'tool-response') and not ("-> Error" in m['content']))]
            # remove print outputs string to save some tokens
            past_observations = "\n".join(past_observations).replace('Print outputs:\n', '')
            formatted_messages = [messages[0],
                                  messages[1],
                                  messages[-2],
                                  {"role": "user",
                                   "content": f"Output of the previous steps:\n {past_observations} \n\n"
                                              + messages[-1]['content']}]
        else:
            formatted_messages = messages.copy()
            for m in formatted_messages:
                if m['role'] == 'tool-response':
                    m['role'] = 'user'
        inputs = self.tokenizer.apply_chat_template(formatted_messages,
                                                    add_generation_prompt=True,
                                                    return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        stopping_criteria = self.get_stopping_criteria(stop_sequences)
        outputs = self.model.generate(**inputs,
                                      **self.generate_kwargs,
                                      stopping_criteria=stopping_criteria,
                                      )
        result = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return result
