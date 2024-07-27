import streamlit as st
import boto3
import json
import yaml
import re
from botocore.exceptions import ClientError

# Load configurations
configs = yaml.load(open(r'D:\Workspace\Learning\llm\streamlit_bedrock\secrets.yaml', 'r'), Loader=yaml.FullLoader)

# AWS credentials setup
AWS_ACCESS_KEY = configs["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = configs["AWS_SECRET_KEY"]
AWS_REGION = configs["AWS_REGION"]

def call_bedrock(prompt):
    # Create an Amazon Bedrock Runtime client.
    brt = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

    # Set the model ID for Titan Text G1 - Lite.
    model_id = "amazon.titan-text-lite-v1"

    # Define the prompt for the model.
    native_request = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.5,
            "topP": 0.9
        },
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = brt.invoke_model(modelId=model_id, body=request)
    except (ClientError, Exception) as e:
        st.error(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return None

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and return the response text.
    return model_response["results"][0]["outputText"]

def extract_media_urls(text):
    # Regular expression to find URLs
    url_pattern = re.compile(r'(https?://\S+)')
    urls = url_pattern.findall(text)
    # Filter out video and image URLs
    video_urls = [url for url in urls if 'youtube.com' in url or 'vimeo.com' in url]
    image_urls = [url for url in urls if url.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    return video_urls, image_urls

# Streamlit app layout
st.title("AWS Bedrock Text Generation")

# Create two columns for layout
col1, col2 = st.columns(2)

with col2:
    st.write("## Enter Prompt")
    prompt = st.text_input("Prompt:")

    if st.button("Generate Text"):
        if prompt:
            with st.spinner("Generating text..."):
                generated_text = call_bedrock(prompt)
        else:
            st.error("Please enter a prompt.")
    else:
        generated_text = ""

with col1:
    st.write("## Preview Generated Text")
    if generated_text:
        st.write(generated_text)
        video_urls, image_urls = extract_media_urls(generated_text)
        for video_url in video_urls:
            st.video(video_url)
        for image_url in image_urls:
            st.image(image_url)
    else:
        st.write("Generated text will appear here.")
