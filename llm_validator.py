import polars as pl
import requests
import json
from typing import Dict, List
import time

class LLMValidator:
    def __init__(self, config: Dict):
        """
        Initialize LLM validator with configuration.
        
        Args:
            config (Dict): Configuration containing:
                ollama_url (str): URL for Ollama API
                model_name (str): Name of the model to use
                primary_keys (List[str]): List of primary key columns   
                batch_size (int): Number of records to process before saving progress
        """
        self.ollama_url = config['ollama_url']
        self.model_name = config.get('model_name', 'llama3.1')
        self.primary_keys = config['primary_keys']
        self.batch_size = config.get('batch_size', 100)
        
    def _create_prompt(self, record: Dict) -> str:
        """Create a prompt for the LLM based on record values."""
        prompt = """Please analyze this insurance record and perform the following tests. 
        Respond with JSON only, containing test results (1 for pass, 0 for fail):
        {
            "ColourTest": 1 or 0,
            "DateConsistencyTest": 1 or 0,
            "PremiumTests": 1 or 0
        }
        
        Tests to perform:
        1. ColourTest: Check if VehicleColour is a real color
        2. DateConsistencyTest: If CoverEndDate is blank then return 1, otherwise verify if CoverEndDate is after CoverStartDate
        3. PremiumTests: Verify MonthlyPremium is positive and less than SumInsured
        
        Record values:
        """
        
        # Add each field and value to the prompt
        for field, value in record.items():
            prompt += f"\n{field}: {value}"
            
        return prompt

    def _call_ollama(self, prompt: str) -> Dict:
        """Make API call to Ollama and parse response."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            
            # Extract JSON from response
            result = response.json()
            response_text = result['response']
            print(f"Full LLM Response for record:\n{response_text}\n")
            
            # Find and parse JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            print(f"Error calling Ollama: {str(e)}")
            # Return failed tests if API call fails
            return {
                "ColourTest": 0,
                "DateConsistencyTest": 0,
                "PremiumTests": 0
            }

    def validate_records(self, data_path: str, output_path: str) -> None:
        """
        Validate records using LLM and save results.
        
        Args:
            data_path (str): Path to input CSV file
            output_path (str): Path to save results CSV
        """
        print("Starting LLM validation...")
        
        # Read data
        df = pl.read_csv(data_path)
        total_records = len(df)
        
        # Initialize results storage
        results = []
        
        # Process records
        for i, record in enumerate(df.iter_rows(named=True)):
            # if i % self.batch_size == 0:
            print(f"Processing record {i+1} of {total_records}")
            
            # Create prompt and get LLM response
            prompt = self._create_prompt(record)
            test_results = self._call_ollama(prompt)
            
            # Prepare result row
            result_row = {key: record[key] for key in self.primary_keys}
            result_row.update(test_results)
            results.append(result_row)
            
            # Save intermediate results periodically
            if (i + 1) % self.batch_size == 0:
                self._save_results(results, output_path, i + 1 == total_records)
                
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        # Save any remaining results
        if results:
            self._save_results(results, output_path, True)
            
        print("LLM validation complete.")

    def _save_results(self, results: List[Dict], output_path: str, is_final: bool) -> None:
        """Save results to CSV file."""
        df = pl.DataFrame(results)
        
        if is_final:
            df.write_csv(output_path)
        else:
            df.write_csv(output_path, mode='append')