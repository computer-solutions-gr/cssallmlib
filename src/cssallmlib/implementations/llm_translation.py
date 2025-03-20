from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
from langchain_ollama import ChatOllama
from typing import Union, Iterator, List
from langchain_core.language_models.chat_models import BaseChatModel


class TranslationChain:
    """A chain for translating text between languages using LLMs."""

    def __init__(self, llm: BaseChatModel, source_lang: str = "English"):
        """Initialize the translation chain.

        Args:
            llm: The language model to use
            source_lang: The source language (defaults to English)
        """
        self.llm = llm
        self.source_lang = source_lang
        self.output_parser = StrOutputParser()

        # Create the translation prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that translates {input_language} to {output_language}. "
                    "Translate the text accurately while preserving the original meaning and tone.",
                ),
                ("human", "{input}"),
            ]
        )

        # Build the chain
        self.chain = self.prompt | self.llm | self.output_parser
        logger.info(f"Initialized TranslationChain with source language: {source_lang}")
        logger.info(f"Streaming capability: {hasattr(self.llm, 'stream')}")

    def translate(
        self, text: str, target_lang: str, stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """Translate text to the target language.

        Args:
            text: The text to translate
            target_lang: The target language
            stream: Whether to stream the translation (if supported by the LLM)

        Returns:
            Either the complete translated text or an iterator of translation chunks

        Raises:
            Exception: If translation fails
        """
        try:
            logger.info(
                f"Starting translation from {self.source_lang} to {target_lang} (streaming={stream})"
            )
            logger.debug(f"Input text: {text[:100]}{'...' if len(text) > 100 else ''}")

            inputs = {
                "input": text,
                "input_language": self.source_lang,
                "output_language": target_lang,
            }

            if stream and hasattr(self.chain, "stream"):
                logger.info("Using streaming translation")
                return self._stream_translate(inputs)
            else:
                if stream:
                    logger.warning(
                        "Streaming requested but not supported by the LLM, falling back to normal translation"
                    )
                result = self.chain.invoke(inputs)
                logger.info(f"Successfully translated text to {target_lang}")
                logger.debug(
                    f"Translated text: {result[:100]}{'...' if len(result) > 100 else ''}"
                )
                return result
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}", exc_info=True)
            raise

    def _stream_translate(self, inputs: dict) -> Iterator[str]:
        """Internal method to handle streaming translation.

        Args:
            inputs: Dictionary containing input text and language information

        Yields:
            Chunks of translated text
        """
        try:
            chunk_count = 0
            for chunk in self.chain.stream(inputs):
                chunk_count += 1
                # logger.debug(f"Streaming chunk {chunk_count}: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
                yield chunk
            logger.info(
                f"Successfully completed streaming translation with {chunk_count} chunks"
            )
        except Exception as e:
            logger.error(f"Streaming translation failed: {str(e)}", exc_info=True)
            raise

    def batch_translate(
        self, texts: List[str], target_lang: str, stream: bool = False
    ) -> Union[List[str], Iterator[List[str]]]:
        """Translate multiple texts to the target language.

        Args:
            texts: List of texts to translate
            target_lang: The target language
            stream: Whether to stream the translations (if supported by the LLM)

        Returns:
            Either a list of translated texts or an iterator of lists of translation chunks

        Raises:
            Exception: If translation fails
        """
        try:
            logger.info(
                f"Starting batch translation of {len(texts)} texts from {self.source_lang} to {target_lang} (streaming={stream})"
            )

            inputs = [
                {
                    "input": text,
                    "input_language": self.source_lang,
                    "output_language": target_lang,
                }
                for text in texts
            ]

            if stream and hasattr(self.chain, "stream"):
                logger.info("Using streaming batch translation")
                return self._stream_batch_translate(inputs)
            else:
                if stream:
                    logger.warning(
                        "Streaming requested but not supported by the LLM, falling back to normal batch translation"
                    )
                results = self.chain.batch(inputs)
                logger.info(
                    f"Successfully completed batch translation of {len(texts)} texts to {target_lang}"
                )
                return results
        except Exception as e:
            logger.error(f"Batch translation failed: {str(e)}", exc_info=True)
            raise

    def _stream_batch_translate(self, inputs: List[dict]) -> Iterator[List[str]]:
        """Internal method to handle streaming batch translation.

        Args:
            inputs: List of dictionaries containing input texts and language information

        Yields:
            Lists of translation chunks
        """
        try:
            chunk_count = 0
            for input in inputs:
                for chunk in self.chain.stream(input):
                    chunk_count += 1
                    # logger.debug(f"Streaming batch chunk {chunk_count}")
                    yield chunk
            logger.info(
                f"Successfully completed streaming batch translation with {chunk_count} chunk sets"
            )
        except Exception as e:
            logger.error(f"Streaming batch translation failed: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    # Initialize the model and translation chain
    llm = ChatOllama(model="llama3.1", temperature=0)
    translator = TranslationChain(llm)

    # Single translation (non-streaming)
    result = translator.translate("I love programming.", "Greek")
    print(f"Translation: {result}")

    # Single translation (streaming)
    print("\nStreaming translation:")
    for chunk in translator.translate("I love programming.", "German", stream=True):
        print(chunk, end="", flush=True)
    print()

    # Batch translation (non-streaming)
    texts = ["Hello world!", "How are you?", "Good morning!"]
    results = translator.batch_translate(texts, "Spanish")
    print("\nBatch translations:")
    for original, translated in zip(texts, results):
        print(f"{original} -> {translated}")

    # Batch translation (streaming)
    print("\nStreaming batch translations:")
    for chunk in translator.batch_translate(texts, "Greek", stream=True):
        print(chunk, end="", flush=True)
    print()
    # for original, chunk in zip(texts, chunks):
    #     print(f"{original} -> {chunk}")
