from ollama import chat
from ollama import ChatResponse

def suminsured_gt_premium(monthly_premium: float, sum_insured: float) -> bool:
    """
    Check if the value of SumInsured is greater than the value of MonthlyPremium.

    Args:
        monthly_premium (float): The value of the monthly premium.
        sum_insured (float): The value of the sum insured.

    Returns:
        bool: True if SumInsured is greater than MonthlyPremium, False otherwise.
    """
    if not isinstance(monthly_premium, (int, float)):
        raise ValueError("monthly_premium must be a number.")
    if not isinstance(sum_insured, (int, float)):
        raise ValueError("sum_insured must be a number.")

    return sum_insured > monthly_premium


messages = [{'role': 'user', 'content': 'If the value of MonthlyPremium is 350 and SumInsured is 350000 is MonthlyPremium greater than SumInsured?'}]
print('Prompt:', messages[0]['content'])

available_functions = {
  'suminsured_gt_premium': suminsured_gt_premium
}

response: ChatResponse = chat(
  'llama3.1',
  messages=messages,
  tools=[suminsured_gt_premium],
)

if response.message.tool_calls:
  # There may be multiple tool calls in the response
  for tool in response.message.tool_calls:
    # Ensure the function is available, and then call it
    if function_to_call := available_functions.get(tool.function.name):
      print('Calling function:', tool.function.name)
      print('Arguments:', tool.function.arguments)
      output = function_to_call(**tool.function.arguments)
      print('Function output:', output)
    else:
      print('Function', tool.function.name, 'not found')

# Only needed to chat with the model using the tool call results
if response.message.tool_calls:
  # Add the function response to messages for the model to use
  messages.append(response.message)
  messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})

  # Get final response from model with function outputs
  final_response = chat('llama3.1', messages=messages)
  print('Final response:', final_response.message.content)

else:
  print('No tool calls returned from model')