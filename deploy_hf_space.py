from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="src/myoquant/streamlit",
    repo_id="corentinm7/MyoQuant",
    repo_type="space",
    ignore_patterns=["*.h5", "*.keras"]
)