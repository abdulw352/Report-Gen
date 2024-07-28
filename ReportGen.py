import os
from typing import List, Dict, Any
from langchain import LLMChain, PromptTemplate
from langchain.llms import Ollama, HuggingFacePipeline
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class ReportGenerator:
    def __init__(self, llm_type: str = "ollama", model_name: str = "llama2"):
        self.llm = self._initialize_llm(llm_type, model_name)
        self.data_source = None
        self.report_elements = []

    def _initialize_llm(self, llm_type: str, model_name: str):
        if llm_type == "ollama":
            return Ollama(model=model_name)
        elif llm_type == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            return HuggingFacePipeline(pipeline=pipe)
        # Add more LLM types as needed
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    def connect_to_data_source(self, source_type: str, **kwargs):
        if source_type == "sql":
            self.data_source = SQLDatabase.from_uri(kwargs['connection_string'])
        elif source_type == "text":
            self.data_source = TextLoader(kwargs['file_path']).load()
        elif source_type == "csv":
            self.data_source = CSVLoader(kwargs['file_path']).load()
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")

    def set_report_elements(self, elements: List[str]):
        self.report_elements = elements

    def generate_report(self, topic: str) -> str:
        if not self.data_source:
            raise ValueError("No data source connected. Use connect_to_data_source() first.")

        if not self.report_elements:
            raise ValueError("No report elements set. Use set_report_elements() first.")

        report_sections = []

        for element in self.report_elements:
            if isinstance(self.data_source, SQLDatabase):
                agent_executor = create_sql_agent(
                    llm=self.llm,
                    toolkit=SQLDatabaseToolkit(db=self.data_source, llm=self.llm),
                    verbose=True
                )
                result = agent_executor.run(
                    f"Generate a report section about {element} related to {topic} using the available data."
                )
            else:
                prompt = PromptTemplate(
                    input_variables=["element", "topic", "data"],
                    template="Generate a report section about {element} related to {topic} using the following data:\n\n{data}"
                )
                chain = LLMChain(llm=self.llm, prompt=prompt)
                result = chain.run(element=element, topic=topic, data=str(self.data_source))

            report_sections.append(result)

        full_report = "\n\n".join(report_sections)
        return full_report

# Example usage
if __name__ == "__main__":
    # Using Ollama
    report_gen_ollama = ReportGenerator(llm_type="ollama", model_name="llama2")
    
    # Using Hugging Face model
    report_gen_hf = ReportGenerator(llm_type="huggingface", model_name="gpt2")  # or any other model available on Hugging Face
    
    # Connect to a SQL database
    report_gen_ollama.connect_to_data_source("sql", connection_string="sqlite:///example.db")
    report_gen_hf.connect_to_data_source("sql", connection_string="sqlite:///example.db")
    
    # Set report elements
    report_elements = ["Introduction", "Key Findings", "Detailed Analysis", "Conclusion"]
    report_gen_ollama.set_report_elements(report_elements)
    report_gen_hf.set_report_elements(report_elements)
    
    # Generate the reports
    report_ollama = report_gen_ollama.generate_report("Market Trends in E-commerce")
    report_hf = report_gen_hf.generate_report("Market Trends in E-commerce")
    
    print("Report generated using Ollama:")
    print(report_ollama)
    print("\nReport generated using Hugging Face model:")
    print(report_hf)
