from typing import Dict, Any, List

from cobot_core import SelectingStrategy
from cobot_core.offensive_speech_classifier import OffensiveSpeechClassifier

import example.sensitive_response_generator.utils as sensitive_response_utils
from example.dialogues.StateTransitionPrompts import StateTransitionPrompts
from example.dialogues.intent_specific_prompt import get_response_for_intent, ask_help_intent
from example.sample_taskbot.global_intent_handler import save_automata_context_in_user_attributes, \
    get_automata_context_from_user_attributes
from example.sensitive_response_generator.filter_response import FilterResponseCreator
from example.state_management.context import ContextFactory
from example.state_management.context import serialize_context
from example.state_management.state import InputAlphabet, EndState, WikiHowState, RecipeQueryState, StateInterface, \
    StackSymbol


class AutomataSelectingStrategy(SelectingStrategy):

    def select_response_mode(self, features: Dict[str, Any]) -> List[str]:
        automata_context = get_automata_context_from_user_attributes(self.state_manager)

        # Reset the graph if we couldn't load it from user attribute above or if we are at end state.
        # Instead of just changing the state, create a new graph so that if backend graph has updated,
        # that is reflected in subsequent requests.
        if automata_context is None or isinstance(automata_context.get_state(), EndState):
            automata_context = ContextFactory().create_context()
            save_automata_context_in_user_attributes(self.state_manager, automata_context)
            # Remove any identified input_alphabet from global intent handler.
            setattr(self.state_manager.current_state, 'input_alphabet', None)

        # Save the automata_context object in current memory so that it can used later.
        setattr(self.state_manager.current_state, 'automata_context', automata_context)

        filter_response_creator = FilterResponseCreator(
            offensive_speech_classifier=OffensiveSpeechClassifier(self.state_manager),
            sensitive_speech_regexes=sensitive_response_utils.get_sensitive_questions_regex(),
            logger=self.logger
        )

        classified_intent = self.get_model_classified_intent()
        text = self.state_manager.current_state.text
        intent = self.state_manager.current_state.intent

        # Get current_three_query_names to remote module at every turn.
        # optionselection module (as well as all remote modules) run at the start of every turn. optionselection remote
        # module uses current_three_query_names to find the best match from the user utterance. The module has access
        # to information for the current and previous turn. current_three_query_names is updated at every turn so that
        # it can be accessed during the next turn.
        if ('recipe_query_result' in self.state_manager.user_attributes.map_attributes
            or 'wikihow_query_result' in self.state_manager.user_attributes.map_attributes) \
                and self.state_manager.user_attributes.query_names:
            top_item_index = self.state_manager.user_attributes.top_item_index
            self.state_manager.current_state.current_three_query_names = \
                self.state_manager.user_attributes.query_names[top_item_index: top_item_index + 3]

        input_alphabet = getattr(self.state_manager.current_state, 'input_alphabet', None)
        add_followup_prompt = False

        # Certain short phrases with 'last' are recognized as AMAZON.PreviousIntent'.
        # For now, if we are in one of the query states, we will select the last item when we
        # recognize such phrases. Once we have intent-classification working we will use a differnt
        # approach as 'go back to the last one' and 'select the last one' requires different actions.
        if (intent == 'general' or intent == 'AMAZON.PreviousIntent') and 'more options' not in text and \
                isinstance(automata_context.get_state(), (WikiHowState, RecipeQueryState)):
            # Saved the matched position based on the name/title of recipe/task to current state.
            # To avoid false positives, we do some checks.

            # If optionselection doesn't run for some reason, the default value will be None. Check to bypass that.
            if self.state_manager.current_state.features['optionselection']:
                match_position = self.state_manager.current_state.features['optionselection'].get('match_position', -1)
                if match_position != -1:
                    self.state_manager.current_state.intent = intent = 'MatchIntent'
                    self.state_manager.current_state.match_position = match_position
            else:
                self.logger.error("Option selection module not working!")

        current_automata_state = automata_context.get_state()
        current_automata_stack_symbol = automata_context.peek_stack()

        if input_alphabet == InputAlphabet.returning_user:
            # This is a returning user with existing task. We will continue from where the user left.
            pass
        else:
            input_alphabet, add_followup_prompt = self.decide_next_graph_input(
                text, intent, classified_intent, filter_response_creator,
                current_automata_state, current_automata_stack_symbol
            )

        # Save the current custom intent (input_alphabet) so that it can be used later.
        setattr(self.state_manager.current_state, 'input_alphabet', input_alphabet)

        transition_done = current_automata_state.handle(input_alphabet, current_automata_stack_symbol, automata_context)
        if transition_done:
            responders = [automata_context.get_state().get_responder_name()]
            setattr(self.state_manager.user_attributes, 'automata_context', serialize_context(automata_context))
        else:
            self.logger.info("Cannot make the transition from {} using input {} and stack symbol {}. ".format(
                current_automata_state, input_alphabet, current_automata_stack_symbol
            ))
            if input_alphabet is InputAlphabet.acknowledgement:
                setattr(
                    self.state_manager.current_state, 'static_response', " "
                )
            elif input_alphabet not in [InputAlphabet.static_response, InputAlphabet.unsafe_task]:
                # when alphabet is static_response alphabet, the static_response variable has already been set,
                # otherwise we set it now with a default message.
                setattr(
                    self.state_manager.current_state,
                    'static_response',
                    StateTransitionPrompts.get_prompt(
                        current_automata_state, current_automata_stack_symbol, self.state_manager
                    )
                )

            if add_followup_prompt:
                followup_prompt = StateTransitionPrompts.get_followup_prompt(
                    current_automata_state, current_automata_stack_symbol, self.state_manager
                )
                new_response = getattr(self.state_manager.current_state, 'static_response', "") + followup_prompt
                setattr(self.state_manager.current_state, 'static_response', new_response)

            responders = ['STATIC_RESPONDER']

        return responders

    def get_model_classified_intent(self) -> str:
        """
        Get the classified intent obtained from remote module if the result exists. Returns empty string otherwise.
        """
        intent_cfn_res = getattr(self.state_manager.current_state, 'intentclassification', None)
        if intent_cfn_res:
            self.logger.info('metric::intent_classification_used')
            return intent_cfn_res['id2label'][str(intent_cfn_res['label'])]
        else:
            self.logger.info('metric::intent_classification_not_used')
            self.logger.warn("intent classification module did not return result.")
            return ""

    def decide_next_graph_input(
            self,
            text: str,
            intent: str,
            classified_intent: str,
            filter_response_creator: FilterResponseCreator,
            current_state: StateInterface,
            stack_symbol: StackSymbol
    ) -> (InputAlphabet, bool):
        """
        Given the text and its classified intents from our model and default Amazon model, gives the input_alphabet
        to use for dialogue state transition, and the need for adding a followup prompt.

        Parameters
        ----------
        current_state:
            current state of the automata. Used for deciding some pre-defined responses.
        stack_symbol:
            current top stack symbol of the automata. Used for deciding some pre-defined responses.
        filter_response_creator:
            Object to filter out user utterance and provide appropriate response string.
        text:
            the input user utterance.
        intent:
            the intent classified using Amazon's default classifier.
        classified_intent:
            the intent classified using the ML remote model.

        Returns
        -------
        Tuple
            (input_alphabet, add_followup_prompt). If it is InputAlphabet.static_response then 'static_response' text
            is also set in the state_manager.current_state.
        """
        task_complete_words = ['complete', 'finish']

        input_alphabet = None
        add_followup_prompt = False

        if intent == 'LaunchRequestIntent':
            input_alphabet = InputAlphabet.launch_phrase
        elif intent == 'AMAZON.CancelIntent':  # Test to see if this works
            input_alphabet = InputAlphabet.cancel
        elif any([keyword in text for keyword in task_complete_words]):
            input_alphabet = InputAlphabet.complete_task
        elif intent in ['SetTimerIntent'] or classified_intent == 'timer':
            input_alphabet = InputAlphabet.timer_task
        elif classified_intent == 'shopping_list':
            input_alphabet = InputAlphabet.list_task
        elif len(filter_response_creator.get_response_for_sensitive_question(text)) > 0:
            static_response = filter_response_creator.last_answer
            setattr(self.state_manager.current_state, 'static_response', static_response)
            add_followup_prompt = filter_response_creator.add_followup_prompt
            if filter_response_creator.end_session:
                input_alphabet = InputAlphabet.unsafe_task
            else:
                input_alphabet = InputAlphabet.static_response
        elif is_acknowledgement(text):
            input_alphabet = InputAlphabet.acknowledgement
        elif 'start cooking' in text:
            input_alphabet = InputAlphabet.start_cooking
        elif is_say_ingredients(text):
            input_alphabet = InputAlphabet.say_ingredients
            setattr(self.state_manager.user_attributes, "say_ingredients", True)
        elif any([keyword in text for keyword in ['resume', 'next', 'previous']]) \
                or intent in ['AMAZON.ResumeIntent', 'AMAZON.NextIntent', 'AMAZON.PreviousIntent']:
            input_alphabet = InputAlphabet.navigation
        elif intent in ['AMAZON.SelectIntent', 'MatchIntent', 'UserEvent']:
            # separate match from any other user intent.
            input_alphabet = InputAlphabet.select_options
        elif intent == 'AMAZON.RepeatIntent' or classified_intent == 'navigation_repeat':
            input_alphabet = InputAlphabet.navigation_repeat
        elif is_ask_help(text) or intent == 'AMAZON.HelpIntent':
            input_alphabet = InputAlphabet.static_response
            help_response = get_response_for_intent(ask_help_intent, current_state, stack_symbol, self.state_manager)
            setattr(self.state_manager.current_state, 'static_response', help_response)

        if not input_alphabet:
            # We don't have intent from remote module. Use regex based classification.
            if len(classified_intent) == 0:
                if any([keyword in text for keyword in ['make', 'bake', 'cook', 'eat', 'drink', 'recipe', 'consume']]):
                    input_alphabet = InputAlphabet.recipe_question
                elif any([phrase in text for phrase in ['how to', 'how do', 'how can']]) or \
                        any([keyword in text for keyword in ['fix', 'repair']]):
                    input_alphabet = InputAlphabet.diy_question
                elif any([keyword in text for keyword in ['how', 'who', 'what', 'which', 'when', 'where', 'why']]):
                    input_alphabet = InputAlphabet.intermediate_question
                elif is_more_options(text):
                    input_alphabet = InputAlphabet.more_options
                elif any(phrase == text for phrase in ['yes', 'sure', 'absolutely', 'definitely', 'fine']):
                    input_alphabet = InputAlphabet.accept
                elif intent in ['NameCaptureIntent']:
                    input_alphabet = InputAlphabet.static_response
                    setattr(self.state_manager.current_state, 'static_response', "NAME_RESPONSE")
                else:
                    input_alphabet = InputAlphabet.undefined
            # We have intent from remote module. Use that for some cases.
            else:
                default_intent_response = get_response_for_intent(
                    classified_intent, current_state, stack_symbol, self.state_manager
                )

                if is_more_options(text):
                    # more_options gets classified as decline so add it before.
                    # this intent will be part of the paraphrasing model due to limited examples.
                    input_alphabet = InputAlphabet.more_options
                elif classified_intent == 'recipe_question':
                    input_alphabet = InputAlphabet.recipe_question
                elif classified_intent == 'intermediate_question':
                    input_alphabet = InputAlphabet.intermediate_question
                elif classified_intent == 'ask_feature_question':
                    input_alphabet = InputAlphabet.static_response
                    setattr(self.state_manager.current_state, 'static_response', default_intent_response)
                elif classified_intent == 'accept':
                    input_alphabet = InputAlphabet.accept
                elif any([phrase in text for phrase in ['how to', 'how do', 'how can']]) or \
                        any([keyword in text for keyword in ['fix', 'repair']]):
                    input_alphabet = InputAlphabet.diy_question
                elif intent in ['NameCaptureIntent']:
                    input_alphabet = InputAlphabet.static_response
                    setattr(self.state_manager.current_state, 'static_response', "NAME_RESPONSE")
                elif len(default_intent_response) > 0:
                    input_alphabet = InputAlphabet.static_response
                    setattr(self.state_manager.current_state, 'static_response', default_intent_response)
                else:
                    input_alphabet = InputAlphabet.undefined

        return input_alphabet, add_followup_prompt


def is_acknowledgement(text: str) -> bool:
    """
    A utility method to tell if the input text should be considered as an acknowledgement phrase or not.
    The text should be valid English phrase without punctuations. This condition is satisfied by the text
    obtained from speech-to-text output.
    """
    sample_phrases = [
        'okay', 'fine', 'cool', 'okay fine', 'i\'m on it', 'nice', 'sure'
    ]
    text = text.strip()
    return any(text == phrase for phrase in sample_phrases)


def is_ask_help(text: str) -> bool:
    """
    A utility method to tell if the input text is asking for help.
    """
    sample_phrases = [
        "help", "help me", "i need help", "i'm lost", "i am lost"
    ]
    text = text.strip()
    return any(text == phrase for phrase in sample_phrases)


def is_more_options(text: str):
    """
    A utility method to determine if we scroll to the next few options.
    """
    sample_phrases = [
        "more options", "none", "neither", "none of these", "neither of these", "none of them", "neither of them",
        "show me more"
    ]

    text = text.strip()
    return any(text == phrase for phrase in sample_phrases)


def is_say_ingredients(text: str) -> bool:
    """
    A utility method to determine if the user said the intent 'say_ingredients'.
    """
    exact_match_phrases = [
        "ingredient", "ingredients"
    ]
    sample_phrases = [
        'say ingredient', 'tell me the ingredient', 'what are the ingredient', 'tell the ingredient',
        'say the ingredient', 'say ingredient', 'tell me ingredient', 'tell ingredient'
    ]
    return any(text == em for em in exact_match_phrases) or any(phrase in text for phrase in sample_phrases)
