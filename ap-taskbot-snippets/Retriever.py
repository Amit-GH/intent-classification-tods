import logging
import re
from abc import abstractmethod, ABC
from cobot_common.service_client import Client
import requests
from requests import HTTPError

from example.service_client.demo_client import client


class Retriever(ABC):
    @abstractmethod
    def __init__(self, params: dict):
        pass

    @abstractmethod
    def retrieve(self, query: str):
        pass

    def _validate_query(self, query):
        assert len(query.strip()) > 0, "Query cannot be empty or blank. "


class Document:
    def __init__(self, title, text, snippet, rank=None, score=None, url=None):
        self.title = title
        self.text = text
        self.snippet = snippet
        self.rank = rank
        self.score = score
        self.url = url

    def __str__(self):
        return "Rank: {}, Score: {}\nTitle: {}\nSnippet: {}\nText: {}\nURL: {}".format(
            self.rank, self.score, self.title, self.snippet, self.text, self.url
        )


class BingRetriever(Retriever):
    def __init__(self, params: dict):
        super(BingRetriever, self).__init__(params)
        self.subscription_key = params['subscription_key']
        self.results_requested_count = params.get('results_requested_count', 10)
        self.search_url = "https://api.bing.microsoft.com/v7.0/search"
        self.headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        self.logger = params['logger']

    def retrieve(self, query: str):
        self._validate_query(query)
        query_params = {
            "q": query,
            "textDecorations": True,
            "textFormat": "HTML",
            "count": self.results_requested_count
        }
        response = requests.get(self.search_url, headers=self.headers, params=query_params)
        try:
            response.raise_for_status()
            search_results = response.json()
        except HTTPError as exp:
            self.logger.exception('Bing search gave error {}. '.format(exp))
            search_results = {}

        # rows = "\n".join(["""<tr>
        #                        <td><a href=\"{0}\">{1}</a></td>
        #                        <td>{2}</td>
        #                      </tr>""".format(v["url"], v["name"], v["snippet"]) \
        #                   for v in search_results["webPages"]["value"]])
        #
        # with open("bing_retriever_result_table.html", "w") as file:
        #     file.write("<table>{0}</table>".format(rows))

        results = []
        for i, search_result in enumerate(search_results.get('webPages', {}).get('value', {})):
            if len(search_result) > 0:
                url = search_result['url']
                title = search_result['name']
                # snippet contains tags which are good for displaying but create issues with SSML output. Remove them.
                snippet = re.sub(r"</?[a-zA-Z]*>", "", search_result['snippet'])
                results.append(Document(title=title, text=None, snippet=snippet, rank=i + 1, url=url))
        return results


class EviRetriever(Retriever):

    def __init__(self, params: dict):
        self.client: Client = params['client']
        self.timeout_in_millis = 2000

    def retrieve(self, query: str):
        self._validate_query(query)
        result = self.client.get_answer(question=query, timeout_in_millis=self.timeout_in_millis)
        response = result['response']
        return [Document(title=None, text=response, snippet=None)]


def user_phrase_for_document(document: Document) -> str:
    # Evi search doesn't return URL and only has text.
    if document.url is None:
        return document.text

    # Bing search gives snippet.
    if document.snippet is not None:
        credit_source = _get_domain(document.url)
        if len(credit_source) == 0:
            credit_source = 'the internet'
        return "According to {}: {}".format(credit_source, document.snippet)

    return ""


def _get_domain(url):
    parts = re.split("/", url)

    match = re.match("(www\.)(\w+)", parts[2])
    if match is not None:
        return match.group(2)

    match = re.match("(\w+\.)(\w+)", parts[2])
    if match is not None:
        return match.group(1) + match.group(2)

    return ''


if __name__ == '__main__':
    bing_params = {
        # Amit's personal free-tier subscription key for testing.
        'subscription_key': '3e7d7e0f38ee47b69f6d73aea785bcb9',
        'results_requested_count': 1,
        'logger': logging.Logger(name='retriever_test_logger')
    }
    bing_retriever = BingRetriever(bing_params)

    evi_params = {'client': client}
    evi_retriever = EviRetriever(evi_params)

    while True:
        search_query = input('entry query to search bing. exit to leave: ')
        if search_query == 'exit':
            break
        bing_result_docs = bing_retriever.retrieve(search_query)
        evi_result_docs = evi_retriever.retrieve(search_query)
        [print(result_doc, '\n') for result_doc in bing_result_docs]
        [print(result_doc, '\n') for result_doc in evi_result_docs]
