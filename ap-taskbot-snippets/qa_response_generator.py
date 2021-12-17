from cobot_core.service_module import ToolkitServiceModule
import requests

from example.dialogues.Prompt import TaskBotPrompts
from example.web_retrieval.Retriever import BingRetriever
from example.web_retrieval.Retriever import user_phrase_for_document

"""
A sample response generator using EVI question answering API with QA classification models.
It calls EVI API to get an answer to an utterance and uses QA classification models to decide whether the response is 
good or not. If Evi response is not confident, we try getting response from Bing search. If that is also
not confident, we return a default response. 
"""

TIMEOUT_IN_MILLIS = 2000
RR_LABEL_GOOD = "1"
RR_LABEL_BAD = "0"
QA_FACTOID_LABEL = "QA_FACTOID_LABEL"
QA_RESPONSE_RELEVANCE_LABEL = "QA_RESPONSE_RELEVANCE_LABEL"

CONFIDENCE_HIGH = 'confidence_high'
CONFIDENCE_MEDIUM = 'confidence_medium'
CONFIDENCE_LOW = 'confidence_low'

# For queries like 'hello', EVI returns a long meaningless string that usually contains a skill ID starting with
# 'skill://'.
# what is the news on corona virus - evi response - Audio: AP News Update COVID-19.mp3
# If any of these are a substring of the EVI response, we discard the response.
# We check for these: (a) ignoring case, (b) ignoring word boundaries (i.e. we just check str.contains()).
RESPONSE_NOT_ALLOWED_SUBSTRINGS = {
    'Alexa,',
    'skill://',
    'Audio:',
    'please say that again',
    'please try that again',
    'catch that',
    'find the answer to the question I heard',
    'I didn’t get that',
    'have an opinion on that',
    'skill',
    'skills',
    'you can',
    'try asking',
    "Here's something I found on",
    'Sorry, I don’t know that'
}

_sessions = requests.Session()


class ResponseGeneratorQA(ToolkitServiceModule):

    def execute(self):
        text = self.state_manager.current_state.text
        evi_response = self._call_evi_service(text)
        is_factoid, relevance_label = self._call_qa_classification_service(text, evi_response)
        evi_response = self.process_response(evi_response)
        confidence = self.get_confidence(evi_response, is_factoid, relevance_label)
        self.logger.info("QA Confidence from Evi: %s", confidence)
        if confidence == CONFIDENCE_HIGH:
            return evi_response

        # Fall back to Bing search as Evi did not give satisfactory results.
        bing_retriever = BingRetriever(params={
            # Amit's personal free-tier subscription key for testing.
            'subscription_key': 'xxxx',
            'results_requested_count': 1,
            'logger': self.logger
        })
        documents = bing_retriever.retrieve(query=text)
        if len(documents) > 0:
            user_phrase = user_phrase_for_document(documents[0])
            is_factoid, relevance_label = self._call_qa_classification_service(text, user_phrase)
            confidence = self.get_confidence(user_phrase, is_factoid, relevance_label)
            self.logger.info("QA Confidence from Bing: %s", confidence)
            if confidence == CONFIDENCE_HIGH:
                return user_phrase

        # Fall back to default answer when no search engine gave good results.
        return TaskBotPrompts.qa_no_answer_found

    @staticmethod
    def create_qa_service_request(text, response):
        request = dict()
        request["turns"] = list()
        request["turns"].append([text, response])
        return request

    def _call_evi_service(self, current_text):
        # Call Evi service
        # Input is a text. Timeout parameter is optional and the client timeout value is used if it's not set.
        try:
            result = self.toolkit_service_client.get_answer(
                question=current_text, timeout_in_millis=TIMEOUT_IN_MILLIS)
            response = result['response']
        except Exception as ex:
            if current_text != '':
                self.logger.exception("An exception while calling QA service",
                                      exc_info=True)
            response = ""
        return response

    def _call_qa_classification_service(self, text, response):
        fc = False
        rr = RR_LABEL_BAD
        request = self.create_qa_service_request(text, response)
        try:
            result = self.toolkit_service_client.get_qa_factoid_response_relevance_results(request)
            fc = result['qa_factoid_classifier_results']['results'][0]
            rr = result['qa_response_relevance_classifier_results']['results']['label']
            self.logger.debug('QA Factoid classifier returned => %s, QA Response Relevance Classifier => %s', fc, rr)
        except Exception as ex:
            self.logger.exception("An exception while calling qa classification service request",
                                  exc_info=True)
        return fc, rr

    @staticmethod
    def get_confidence(evi_response, is_factoid, relevance_label):
        """
        Get confidence about the appropriateness of EVI response. In real use case, it can be modified to return
        a numeric value that can be used for ranking by custom ranking strategy.
        """
        confidence = CONFIDENCE_LOW
        if evi_response and is_factoid:
            confidence = CONFIDENCE_HIGH
        elif evi_response and not is_factoid and relevance_label == RR_LABEL_GOOD:
            confidence = CONFIDENCE_HIGH
        elif evi_response:
            confidence = CONFIDENCE_MEDIUM
        return confidence

    @staticmethod
    def process_response(response):
        """
        :param response: check if response contains any not_allowed phrase. Evi qa api sometimes returns not_allowed responses.
         We don't want to use them
        :return:
        """
        if response and any(string.lower() in response.lower() for string in RESPONSE_NOT_ALLOWED_SUBSTRINGS):
            return ""

        return response if response else ''
