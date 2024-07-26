import os
from typing import List, Dict, Any
from langchain import LLMChain, PromptTemplate
from langchain.llms import Ollama
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

class ReportGenerator:
    def __init__(self, llm_type: str = "ollama", model_name: str = "llama2"):
        self.llm = self._initialize_llm(llm_type, model_name)
        self.data_source = None
        self.report_elements = []

    def _initialize_llm(self, llm_type: str, model_name: str):
        if llm_type == "ollama":
            return Ollama(model=model_name)
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
    report_gen = ReportGenerator(llm_type="ollama", model_name="llama2")
    
    # Connect to a SQL database
    report_gen.connect_to_data_source("sql", connection_string="sqlite:///example.db")
    
    # Set report elements
    report_gen.set_report_elements(["Introduction", "Key Findings", "Detailed Analysis", "Conclusion"])
    
    # Generate the report
    report = report_gen.generate_report("Market Trends in E-commerce")
    
    print(report)
