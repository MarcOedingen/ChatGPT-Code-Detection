import tiktoken
import Utility.utils as utils


class Tokenizer:
    def __init__(self):
        self._encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _encode(self, message):
        return self._encoding.encode(message)

    def _decode(self, message):
        return self._encoding.decode(message)

    def _count_tokens_messages(self, messages):
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self._encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _count_tokens_code(self, code):
        return len(self._encode(code))

    def _tokenize(self, code):
        return self._encode(code)
