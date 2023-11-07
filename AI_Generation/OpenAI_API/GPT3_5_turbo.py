import os
import openai


class GPT3_turbo:
    def __init__(self):
        self._set_key()
        self._set_organization()

    def _query(self, message_log, max_tokens, temperature):
        attempt = 0
        max_attempts = 3
        while attempt <= max_attempts:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message_log,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response
            except Exception as e:
                attempt += 1
                print(e)
                print(f" Attempt {attempt+1}: Error in API call. Please try again.")
                continue
        print("Error in API call. Please try again.")
        return None

    def _set_key(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def _get_key(self):
        return openai.api_key

    def _set_organization(self):
        openai.organization = os.environ.get("ORGANIZATION_ID")

    def _get_organization(self):
        return openai.organization
