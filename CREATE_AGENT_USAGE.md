# Create Agent - Usage Guide

This guide shows how to create an e-commerce chat assistant agent using the provided CSV file and prompt.

## Files Provided

1. **mock_data.csv** - Contains e-commerce order data with order status, customer information, and ticket data
2. **agent_prompt.txt** - Contains the complete prompt with tool usage instructions and examples
3. **insert_tools.sql** - SQL script to insert the required tools directly into the database

## API Endpoint

```
POST /agents/
```

## Request Format

The endpoint accepts a multipart form-data request with:
- `prompt` (string, required): The agent prompt
- `csv_file` (file, required): CSV file containing knowledge base data
- `tool_ids` (string, optional): Comma-separated list of tool UUIDs

## Example Request using cURL

```bash
curl -X POST "http://localhost:8000/agents/" \
  -F "prompt=$(cat agent_prompt.txt)" \
  -F "csv_file=@mock_data.csv" \
  -F "tool_ids=your-tool-id-1,your-tool-id-2"
```

## Example Request using Python (requests)

```python
import requests

# Read the prompt file
with open('agent_prompt.txt', 'r') as f:
    prompt = f.read()

# Prepare the request
url = "http://localhost:8000/agents/"
files = {
    'csv_file': ('mock_data.csv', open('mock_data.csv', 'rb'), 'text/csv')
}
data = {
    'prompt': prompt,
    'tool_ids': 'your-tool-id-1,your-tool-id-2'  # Optional, comma-separated
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Creating Tools First

Before creating the agent, you'll need to create the two tools (`get-order-status` and `create-a-ticket`). You can do this either via the API or directly in the database.

### Option 1: Using SQL Script (Recommended)

Execute the provided SQL script directly against your PostgreSQL database:

```bash
psql -h localhost -U your_username -d your_database -f insert_tools.sql
```

Or if using a connection string:
```bash
psql "postgresql://user:password@localhost/dbname" -f insert_tools.sql
```

The script will:
- Insert both tools with auto-generated UUIDs
- Use `ON CONFLICT` to update if tools already exist (based on tool_name)
- Display the tool IDs after insertion for easy reference

After running the script, retrieve the tool IDs:
```sql
SELECT tool_id, tool_name FROM tools WHERE tool_name IN ('get-order-status', 'create-a-ticket');
```

### Option 2: Using API Endpoints

#### Create get-order-status Tool

```bash
curl -X POST "http://localhost:8000/tools/" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "get-order-status",
    "description": "Retrieves order status information for customers",
    "parameters": {
      "curl": "curl -X GET \"https://api.example.com/orders/{order_number}\" -H \"Authorization: Bearer API_KEY\" -H \"Content-Type: application/json\""
    }
  }'
```

#### Create create-a-ticket Tool

```bash
curl -X POST "http://localhost:8000/tools/" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "create-a-ticket",
    "description": "Creates a support ticket when customers have problems or issues",
    "parameters": {
      "curl": "curl -X POST \"https://api.example.com/tickets\" -H \"Authorization: Bearer API_KEY\" -H \"Content-Type: application/json\" -d '\''{\"order_number\": \"{order_number}\", \"customer_email\": \"{customer_email}\", \"issue_type\": \"{issue_type}\", \"issue_description\": \"{issue_description}\", \"priority\": \"{priority}\"}'\''"
    }
  }'
```

**Note:** Replace `https://api.example.com` with your actual API endpoint. The `{variable}` placeholders in the curl commands will need to be substituted with actual values when the tool is executed.

## CSV File Structure

The `mock_data.csv` file contains the following columns:
- `order_id`: Unique order identifier
- `order_number`: Order number used by customers
- `customer_name`: Customer name
- `email`: Customer email
- `order_date`: Date order was placed
- `order_status`: Current order status (Pending, Processing, Shipped, Delivered, Cancelled)
- `shipping_status`: Shipping status (Not Shipped, Preparing, Shipped, In Transit, Delivered, Cancelled)
- `delivery_date`: Expected or actual delivery date
- `product_name`: Name of the product ordered
- `product_quantity`: Quantity ordered
- `product_price`: Price per unit
- `total_amount`: Total order amount
- `shipping_address`: Delivery address
- `tracking_number`: Shipping tracking number
- `payment_status`: Payment status (Pending, Paid, Refunded)
- `issue_type`: Type of issue (if applicable)
- `issue_description`: Description of the issue (if applicable)
- `ticket_id`: Support ticket ID (if applicable)
- `ticket_status`: Ticket status (Open, In Progress, Resolved, Closed)
- `resolution`: Resolution details (if applicable)

## Prompt Overview

The prompt file (`agent_prompt.txt`) includes:
1. Agent role description
2. Tool definitions and usage instructions
3. Mock API calls for each tool
4. Guidelines for when and how to use each tool
5. Example conversations
6. Error handling instructions

## Expected Response

On success, the API will return:
- Agent ID
- Namespace (derived from CSV filename: `mock_data`)
- Prompt
- Associated tool IDs
- Created timestamp

## Testing the Agent

After creating the agent, you can test it by:
1. Making a phone call to the Twilio number associated with the agent
2. Sending messages through the WebSocket connection
3. Querying the agent API endpoints

The agent will be able to:
- Retrieve order status information using the `get-order-status` tool
- Create support tickets using the `create-a-ticket` tool
- Answer questions based on the knowledge base from the CSV file
