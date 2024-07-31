import os
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from langchain import LLMChain, PromptTemplate
from langchain.llms import Ollama, HuggingFacePipeline
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.indices.struct_store import SQLStructStoreIndex
from llama_index.objects import SQLDataSource

class ReportGenerator:
    def __init__(self, llm_type: str = "ollama", model_name: str = "llama2"):
        self.llm = self._initialize_llm(llm_type, model_name)
        self.data_source = None
        self.report_elements = []
        self.df = None
        self.index = None

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
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    def connect_to_data_source(self, source_type: str, **kwargs):
        if source_type == "sql":
            self.data_source = SQLDatabase.from_uri(kwargs['connection_string'])
            self.df = pd.read_sql(kwargs['query'], self.data_source.engine)
            sql_data_source = SQLDataSource(self.data_source.engine)
            llm_predictor = LLMPredictor(llm=self.llm)
            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
            self.index = SQLStructStoreIndex(
                sql_data_source, service_context=service_context
            )
        elif source_type == "csv":
            self.df = pd.read_csv(kwargs['file_path'])
            documents = SimpleDirectoryReader(input_files=[kwargs['file_path']]).load_data()
            self.index = GPTVectorStoreIndex.from_documents(documents)
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")

    def set_report_elements(self, elements: List[str]):
        self.report_elements = elements

    def generate_report(self, topic: str) -> str:
        if not self.data_source and self.df is None:
            raise ValueError("No data source connected. Use connect_to_data_source() first.")

        if not self.report_elements:
            raise ValueError("No report elements set. Use set_report_elements() first.")

        report_sections = []

        for element in self.report_elements:
            query = f"Generate a report section about {element} related to {topic} using the available data."
            response = self.index.query(query)
            report_sections.append(str(response))

            # Generate a visualization
            fig = self._generate_visualization(element)
            if fig:
                fig.savefig(f"{element.lower().replace(' ', '_')}.png")
                plt.close(fig)

        full_report = "\n\n".join(report_sections)
        return full_report

    def _generate_visualization(self, element: str):
        if self.df is None:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        if element.lower() == "key findings":
            self.df.plot(kind='bar', ax=ax)
            ax.set_title(f"Bar Plot for {element}")
        elif element.lower() == "detailed analysis":
            self.df.plot(kind='line', ax=ax)
            ax.set_title(f"Line Plot for {element}")
        else:
            self.df.plot(kind='scatter', x=self.df.columns[0], y=self.df.columns[1], ax=ax)
            ax.set_title(f"Scatter Plot for {element}")

        return fig

    def save_report_as_pdf(self, report: str, filename: str):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for element, content in zip(self.report_elements, report.split("\n\n")):
            pdf.cell(200, 10, txt=element, ln=1, align='C')
            pdf.multi_cell(0, 10, txt=content)
            
            image_path = f"{element.lower().replace(' ', '_')}.png"
            if os.path.exists(image_path):
                pdf.image(image_path, x=10, w=190)

        pdf.output(filename)

# Example usage
if __name__ == "__main__":
    report_gen = ReportGenerator(llm_type="ollama", model_name="llama2")
    
    # Connect to a SQL database
    report_gen.connect_to_data_source("sql", connection_string="sqlite:///example.db", query="SELECT * FROM your_table")
    
    # Set report elements
    report_gen.set_report_elements(["Introduction", "Key Findings", "Detailed Analysis", "Conclusion"])
    
    # Generate the report
    report = report_gen.generate_report("Market Trends in E-commerce")
    
    # Save the report as PDF
    report_gen.save_report_as_pdf(report, "market_trends_report.pdf")
    
    print("Report generated and saved as 'market_trends_report.pdf'")
