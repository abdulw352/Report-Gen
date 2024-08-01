import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
# ... (other imports remain the same)

class GenerateReport:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_summary_stats(self) -> Dict[str, Any]:
        return {
            'total_rows': len(self.df),
            'columns': list(self.df.columns),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist(),
            'date_columns': self.df.select_dtypes(include=['datetime64']).columns.tolist(),
            'summary': self.df.describe().to_dict()
        }

    def get_correlation_matrix(self) -> pd.DataFrame:
        return self.df.corr()

    def generate_time_series_plot(self, date_column: str, value_column: str, title: str) -> str:
        plt.figure(figsize=(10, 6))
        plt.plot(self.df[date_column], self.df[value_column])
        plt.title(title)
        plt.xlabel(date_column)
        plt.ylabel(value_column)
        filename = f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()
        return filename

    def generate_bar_plot(self, x_column: str, y_column: str, title: str) -> str:
        plt.figure(figsize=(10, 6))
        self.df.plot(x=x_column, y=y_column, kind='bar')
        plt.title(title)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        filename = f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()
        return filename

    def get_top_n_rows(self, column: str, n: int, ascending: bool = False) -> pd.DataFrame:
        return self.df.sort_values(by=column, ascending=ascending).head(n)

    def calculate_growth_rate(self, date_column: str, value_column: str) -> float:
        first_value = self.df.sort_values(date_column).iloc[0][value_column]
        last_value = self.df.sort_values(date_column).iloc[-1][value_column]
        time_diff = (self.df[date_column].max() - self.df[date_column].min()).days / 365.25  # in years
        return ((last_value / first_value) ** (1 / time_diff) - 1) * 100  # annualized growth rate

class ReportGenerator:
    def __init__(self, llm_config: Dict[str, Dict[str, str]]):
        self.llms = self._initialize_llms(llm_config)
        self.data_source = None
        self.report_elements = []
        self.df = None
        self.index = None
        self.generate_report_instance = None

    # ... (other methods remain the same)

    def connect_to_data_source(self, source_type: str, **kwargs):
        # ... (previous implementation remains)
        if self.df is not None:
            self.generate_report_instance = GenerateReport(self.df)

    def generate_report(self, topic: str) -> str:
        if self.df is None or self.generate_report_instance is None:
            raise ValueError("No data source connected. Use connect_to_data_source() first.")

        if not self.report_elements:
            raise ValueError("No report elements set. Use set_report_elements() first.")

        tools = [
            Tool(
                name="Pandas DataFrame Query",
                func=lambda q: self.index.query(q),
                description="Useful for querying the pandas DataFrame to get information for the report."
            ),
            Tool(
                name="Generate Report Functions",
                func=lambda func_name, *args, **kwargs: getattr(self.generate_report_instance, func_name)(*args, **kwargs),
                description="Useful for calling pre-written functions to analyze data and create visualizations."
            ),
            Tool(
                name="Python REPL",
                func=PythonREPLTool().run,
                description="Useful for running Python code to analyze data or create visualizations when pre-written functions are not sufficient."
            ),
        ]

        prompt = CustomPromptTemplate(
            template="""You are an AI assistant tasked with generating a comprehensive report. Your goal is to create a well-structured, informative report based on the given topic and data.

Available tools:
{tools}

You have access to a GenerateReport class with the following pre-written functions:
- get_summary_stats(): Returns a dictionary with summary statistics of the DataFrame.
- get_correlation_matrix(): Returns the correlation matrix of numeric columns.
- generate_time_series_plot(date_column, value_column, title): Generates a time series plot and returns the filename.
- generate_bar_plot(x_column, y_column, title): Generates a bar plot and returns the filename.
- get_top_n_rows(column, n, ascending=False): Returns the top n rows sorted by the specified column.
- calculate_growth_rate(date_column, value_column): Calculates the annualized growth rate.

Use these functions whenever possible instead of writing custom code. Only use the Python REPL tool if you need functionality not provided by these functions.

Use the following format:

Topic: the topic of the report
Thought: consider what needs to be done to create the report
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have all the information to write the report
Final Answer: the complete report

Begin!

Topic: {topic}
{agent_scratchpad}""",
            tools=tools,
        )

        # ... (rest of the generate_report method remains the same)

# Example usage
if __name__ == "__main__":
    llm_config = {
        'task_master': {'type': 'ollama', 'model': 'llama2'},
        'writer': {'type': 'ollama', 'model': 'llama2'},
        'analyst': {'type': 'ollama', 'model': 'llama2'},
        'programmer': {'type': 'ollama', 'model': 'codellama'}
    }
    
    report_gen = ReportGenerator(llm_config)
    
    # Create a sample DataFrame
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'Sales': [100, 120, 80, 140, 160, 190, 210, 230, 200, 180, 160, 150],
        'Customers': [50, 55, 45, 60, 70, 80, 85, 90, 85, 75, 70, 65]
    })
    
    # Connect to the DataFrame as a data source
    report_gen.connect_to_data_source("dataframe", dataframe=df)
    
    # Set report elements
    report_gen.set_report_elements(["Introduction", "Key Findings", "Detailed Analysis", "Conclusion"])
    
    # Generate the report
    report = report_gen.generate_report("E-commerce Sales Trends in 2023")
    
    # Save the report as PDF
    report_gen.save_report_as_pdf(report, "ecommerce_sales_report_2023.pdf")
    
    print("Report generated and saved as 'ecommerce_sales_report_2023.pdf'")
