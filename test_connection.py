from dotenv import load_dotenv
import os

# Force-load .env from the current directory
load_dotenv()

from justicelens.watsonx_integration import GraniteClient

client = GraniteClient()
print(f"Configured: {client.is_available()}")

if client.is_available():
    result = client.generate_safe("Hello, are you working?")
    if result:
        print("✅ SUCCESS! Granite replied:")
        print(result.text[:200])
    else:
        print("❌ Generation failed. Check your credentials.")
else:
    print("❌ Credentials missing or invalid. Check your .env file.")