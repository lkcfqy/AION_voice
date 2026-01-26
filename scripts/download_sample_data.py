import os
import urllib.request
import tarfile

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    url = "http://www.openslr.org/resources/31/dev-clean-2.tar.gz"
    filename = "dev-clean-2.tar.gz"
    filepath = os.path.join(data_dir, filename)
    
    print(f"Downloading Mini LibriSpeech sample to {data_dir}...")
    print(f"URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        print("Download complete. Extracting...")
        
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=data_dir)
            
        print("Extraction complete.")
        print(f"Data ready at: {os.path.join(data_dir, 'LibriSpeech')}")
        
    except Exception as e:
        print(f"Failed: {e}")
        print("Please download manually or check internet connection.")

if __name__ == "__main__":
    main()
