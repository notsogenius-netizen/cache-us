-- SQL Script to insert raise-a-ticket tool into the database
-- Table: tools
-- 
-- Execute this script against your database to create the raise-a-ticket tool
-- This tool creates support tickets by posting to http://localhost:3000/create-a-ticket

-- Insert raise-a-ticket tool
INSERT INTO tools (tool_id, tool_name, description, parameters, created_at)
VALUES (
    gen_random_uuid(),  -- Generates a random UUID for tool_id
    'raise-a-ticket',
    'Creates a support ticket by posting to http://localhost:3000/raise-a-ticket. Use this whenever the user reports any issue. The reason should be a clear and descriptive summary of the problem the user is facing.',
    '{"curl": "curl -X POST \"http://localhost:3000/raise-a-ticket\" -H \"Content-Type: application/json\" -d '\''{}'\''"}'::jsonb,
    NOW()
)
ON CONFLICT (tool_name) DO UPDATE SET
    description = EXCLUDED.description,
    parameters = EXCLUDED.parameters;

-- Query to retrieve the inserted tool ID (useful for creating the agent)
SELECT tool_id, tool_name, description
FROM tools
WHERE tool_name = 'raise-a-ticket';

