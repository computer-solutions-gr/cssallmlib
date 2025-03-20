from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
from langchain_ollama import ChatOllama

class TranslationChain:
    """A chain for translating text between languages using LLMs."""
    
    def __init__(self, llm, source_lang: str = "English"):
        """Initialize the translation chain.
        
        Args:
            llm: The language model to use
            source_lang: The source language (defaults to English)
        """
        self.llm = llm
        self.source_lang = source_lang
        self.output_parser = StrOutputParser()
        
        # Create the translation prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that translates {input_language} to {output_language}. "
             "Translate the text accurately while preserving the original meaning and tone."),
            ("human", "{input}")
        ])
        
        # Build the chain
        self.chain = (
            {"input": RunnablePassthrough(), 
             "input_language": lambda x: self.source_lang,
             "output_language": lambda x: x["target_lang"]} 
            | self.prompt 
            | self.llm 
            | self.output_parser
        )
        logger.info(f"Initialized TranslationChain with source language: {source_lang}")
    
    def translate(self, text: str, target_lang: str) -> str:
        """Translate text to the target language.
        
        Args:
            text: The text to translate
            target_lang: The target language
            
        Returns:
            The translated text
            
        Raises:
            Exception: If translation fails
        """
        try:
            logger.info(f"Starting translation from {self.source_lang} to {target_lang}")
            logger.debug(f"Input text: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            result = self.chain.invoke({"input": text, "target_lang": target_lang})
            
            logger.info(f"Successfully translated text to {target_lang}")
            logger.debug(f"Translated text: {result[:100]}{'...' if len(result) > 100 else ''}")
            return result
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}", exc_info=True)
            raise

    def batch_translate(self, texts: list[str], target_lang: str) -> list[str]:
        """Translate multiple texts to the target language.
        
        Args:
            texts: List of texts to translate
            target_lang: The target language
            
        Returns:
            List of translated texts
        """
        try:
            logger.info(f"Starting batch translation of {len(texts)} texts from {self.source_lang} to {target_lang}")
            
            results = self.chain.batch([
                {"input": text, "target_lang": target_lang} 
                for text in texts
            ])
            
            logger.info(f"Successfully completed batch translation of {len(texts)} texts to {target_lang}")
            return results
        except Exception as e:
            logger.error(f"Batch translation failed: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    # Initialize the model and translation chain
    llm = ChatOllama(model="llama3.1", temperature=0)
    translator = TranslationChain(llm)

    # Single translation
    result = translator.translate("I love programming.", "German")
    print(f"Translation: {result}")

    # Batch translation
    texts = ["Hello world!", "How are you?", "Good morning!"]
    results = translator.batch_translate(texts, "Spanish")
    print("\nBatch translations:")
    for original, translated in zip(texts, results):
        print(f"{original} -> {translated}")
