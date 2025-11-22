-- SQL Script to insert e-commerce tools into the database
-- Table: tools
-- 
-- Execute this script against your database to create the get-order-status and create-a-ticket tools

-- Insert get-order-status tool
INSERT INTO tools (tool_id, tool_name, description, parameters, created_at)
VALUES (
    gen_random_uuid(),  -- Generates a random UUID for tool_id
    'get-order-status',
    'Retrieves order status information for customers',
    '{"curl": "curl -X GET \"https://api.example.com/orders/{order_number}\" -H \"Authorization: Bearer API_KEY\" -H \"Content-Type: application/json\""}'::jsonb,
    NOW()
)
ON CONFLICT (tool_name) DO UPDATE SET
    description = EXCLUDED.description,
    parameters = EXCLUDED.parameters;

-- Insert create-a-ticket tool
INSERT INTO tools (tool_id, tool_name, description, parameters, created_at)
VALUES (
    gen_random_uuid(),  -- Generates a random UUID for tool_id
    'create-a-ticket',
    'Creates a support ticket when customers have problems or issues',
    '{"curl": "curl -X POST \"https://api.example.com/tickets\" -H \"Authorization: Bearer API_KEY\" -H \"Content-Type: application/json\" -d '\''{\"order_number\": \"{order_number}\", \"customer_email\": \"{customer_email}\", \"issue_type\": \"{issue_type}\", \"issue_description\": \"{issue_description}\", \"priority\": \"{priority}\"}'\''"}'::jsonb,
    NOW()
)
ON CONFLICT (tool_name) DO UPDATE SET
    description = EXCLUDED.description,
    parameters = EXCLUDED.parameters;

-- Query to retrieve the inserted tool IDs (useful for creating the agent)
SELECT tool_id, tool_name, description
FROM tools
WHERE tool_name IN ('get-order-status', 'create-a-ticket')
ORDER BY tool_name;

