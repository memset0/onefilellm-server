# Forked from jimmc414/onefilellm @ f92ef3d

import requests
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader
import os
import sys
import tiktoken
import nltk
from nltk.corpus import stopwords
import re
import nbformat
from nbconvert import PythonExporter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import pyperclip
import wget
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import xml.etree.ElementTree as ET  # Keep for preprocess_text if needed

# --- Configuration Flags ---
ENABLE_COMPRESSION_AND_NLTK = False  # Set to True to enable NLTK download, stopword removal, and compressed output
# --- End Configuration Flags ---

# --- Output Format Notes ---
# This script produces output wrapped in XML-like tags for structure (e.g., <source>, <file>).
# However, the *content* within these tags (especially code) is NOT XML-escaped.
# This means characters like < > & within code blocks are preserved as-is for readability
# and correct interpretation by LLMs. The escape_xml function currently returns text unchanged.
# --- End Output Format Notes ---

EXCLUDED_DIRS = ["dist", "node_modules", ".git", "__pycache__"]


def safe_file_read(filepath, fallback_encoding='latin1'):
    try:
        with open(filepath, "r", encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding=fallback_encoding) as file:
            return file.read()


stop_words = set()
if ENABLE_COMPRESSION_AND_NLTK:
    try:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
    except Exception as e:
        print(f"[bold yellow]Warning:[/bold yellow] Failed to download or load NLTK stopwords. Compression will proceed without stopword removal. Error: {e}")

TOKEN = os.getenv('GITHUB_TOKEN', 'default_token_here')
if TOKEN == 'default_token_here':
    # Consider making this a non-fatal warning or prompt if interactive use is common
    print("[bold red]Warning:[/bold red] GITHUB_TOKEN environment variable not set. GitHub API requests may fail or be rate-limited.")
    # raise EnvironmentError("GITHUB_TOKEN environment variable not set.") # Keep commented out if you want it to proceed

headers = {"Authorization": f"token {TOKEN}"} if TOKEN != 'default_token_here' else {}


def download_file(url, target_path):
    # Add headers conditionally
    response = requests.get(url, headers=headers if TOKEN != 'default_token_here' else None)
    response.raise_for_status()
    with open(target_path, "wb") as f:
        f.write(response.content)


def process_ipynb_file(temp_file):
    try:
        with open(temp_file, "r", encoding='utf-8', errors='ignore') as f:
            notebook_content = f.read()
        exporter = PythonExporter()
        python_code, _ = exporter.from_notebook_node(nbformat.reads(notebook_content, as_version=4))
        return python_code
    except Exception as e:
        print(f"[bold red]Error processing notebook {temp_file}: {e}[/bold red]")
        # Return error message instead of raising, wrapped in comments
        return f"# ERROR PROCESSING NOTEBOOK: {e}\n"


# --- XML Handling ---
def escape_xml(text):
    """
    Returns the text unchanged.
    This function previously escaped XML special characters (&, <, >),
    but that made code content unreadable. It's kept for potential future use
    (e.g., escaping attributes if needed) but currently acts as a pass-through
    to ensure code readability within XML tags.
    """
    return str(text)
# --- End XML Handling ---


def process_github_repo(repo_url):
    """
    Processes a GitHub repository, extracting file contents and wrapping them in XML structure.
    """
    api_base_url = "https://api.github.com/repos/"
    repo_url_parts = repo_url.split("https://github.com/")[-1].split("/")
    repo_name = "/".join(repo_url_parts[:2])
    branch_or_tag = ""
    subdirectory = ""

    if len(repo_url_parts) > 2 and repo_url_parts[2] == "tree":
        if len(repo_url_parts) > 3:
            branch_or_tag = repo_url_parts[3]
        if len(repo_url_parts) > 4:
            subdirectory = "/".join(repo_url_parts[4:])

    contents_url = f"{api_base_url}{repo_name}/contents"
    if subdirectory:
        contents_url = f"{contents_url}/{subdirectory}"
    if branch_or_tag:
        contents_url = f"{contents_url}?ref={branch_or_tag}"

    # Start XML structure
    repo_content = [f'<source type="github_repository" url="{escape_xml(repo_url)}">']

    def process_directory_recursive(url, repo_content_list):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            files = response.json()

            for file_info in files:
                if file_info["type"] == "dir" and file_info["name"] in EXCLUDED_DIRS:
                    continue

                if file_info["type"] == "file" and is_allowed_filetype(file_info["name"]):
                    print(f"Processing {file_info['path']}...")
                    temp_file = f"temp_{file_info['name']}"
                    try:
                        download_file(file_info["download_url"], temp_file)
                        repo_content_list.append(f'\n<file path="{escape_xml(file_info["path"])}">')
                        if file_info["name"].endswith(".ipynb"):
                            # Append raw code - escape_xml not needed as it does nothing
                            repo_content_list.append(process_ipynb_file(temp_file))
                        else:
                            # Append raw code - escape_xml not needed here
                            repo_content_list.append(safe_file_read(temp_file))
                        repo_content_list.append('</file>')
                    except Exception as e:
                        print(f"[bold red]Error processing file {file_info['path']}: {e}[/bold red]")
                        repo_content_list.append(f'\n<file path="{escape_xml(file_info["path"])}">')
                        repo_content_list.append(f'<error>Failed to download or process: {escape_xml(str(e))}</error>')
                        repo_content_list.append('</file>')
                    finally:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                elif file_info["type"] == "dir":
                    process_directory_recursive(file_info["url"], repo_content_list)
        except requests.exceptions.RequestException as e:
            print(f"[bold red]Error fetching directory {url}: {e}[/bold red]")
            repo_content_list.append(f'<error>Failed to fetch directory {escape_xml(url)}: {escape_xml(str(e))}</error>')
        except Exception as e:  # Catch other potential errors like JSON parsing
            print(f"[bold red]Error processing directory {url}: {e}[/bold red]")
            repo_content_list.append(f'<error>Failed processing directory {escape_xml(url)}: {escape_xml(str(e))}</error>')

    process_directory_recursive(contents_url, repo_content)
    repo_content.append('\n</source>')  # Close source tag
    print("GitHub repository processing finished.")
    return "\n".join(repo_content)


def process_local_folder(local_path):
    """
    Processes a local directory, extracting file contents and wrapping them in XML structure.
    """
    def process_local_directory_recursive(current_path, content_list):
        try:
            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                relative_path = os.path.relpath(item_path, local_path)

                if os.path.isdir(item_path):
                    if item not in EXCLUDED_DIRS:
                        process_local_directory_recursive(item_path, content_list)
                elif os.path.isfile(item_path):
                    if is_allowed_filetype(item):
                        print(f"Processing {item_path}...")
                        content_list.append(f'\n<file path="{escape_xml(relative_path)}">')
                        try:
                            if item.endswith(".ipynb"):
                                content_list.append(process_ipynb_file(item_path))
                            else:
                                content_list.append(safe_file_read(item_path))
                        except Exception as e:
                            print(f"[bold red]Error reading file {item_path}: {e}[/bold red]")
                            content_list.append(f'<error>Failed to read file: {escape_xml(str(e))}</error>')
                        content_list.append('</file>')
        except Exception as e:
            print(f"[bold red]Error reading directory {current_path}: {e}[/bold red]")
            content_list.append(f'<error>Failed reading directory {escape_xml(current_path)}: {escape_xml(str(e))}</error>')

    # Start XML structure
    content = [f'<source type="local_folder" path="{escape_xml(local_path)}">']
    process_local_directory_recursive(local_path, content)
    content.append('\n</source>')  # Close source tag

    print("Local folder processing finished.")
    return '\n'.join(content)


def process_arxiv_pdf(arxiv_abs_url):
    """
    Downloads and extracts text from an ArXiv PDF, wrapped in XML.
    """
    pdf_url = arxiv_abs_url.replace("/abs/", "/pdf/") + ".pdf"
    temp_pdf_path = 'temp_arxiv.pdf'
    try:
        print(f"Downloading ArXiv PDF from {pdf_url}...")
        response = requests.get(pdf_url)
        response.raise_for_status()

        with open(temp_pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)

        print("Extracting text from PDF...")
        text_list = []
        with open(temp_pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for i, page in enumerate(range(len(pdf_reader.pages))):
                print(f"  Processing page {i+1}/{len(pdf_reader.pages)}")
                page_text = pdf_reader.pages[page].extract_text()
                if page_text:  # Add text only if extraction was successful
                    text_list.append(page_text)

        # Use XML structure
        formatted_text = f'<source type="arxiv" url="{escape_xml(arxiv_abs_url)}">\n'
        formatted_text += ' '.join(text_list)  # Append raw extracted text
        formatted_text += '\n</source>'  # Close source tag
        print("ArXiv paper processed successfully.")
        return formatted_text

    except requests.RequestException as e:
        print(f"[bold red]Error downloading ArXiv PDF {pdf_url}: {e}[/bold red]")
        return f'<source type="arxiv" url="{escape_xml(arxiv_abs_url)}"><error>Failed to download PDF: {escape_xml(str(e))}</error></source>'
    except Exception as e:  # Catch PdfReader errors or others
        print(f"[bold red]Error processing ArXiv PDF {arxiv_abs_url}: {e}[/bold red]")
        return f'<source type="arxiv" url="{escape_xml(arxiv_abs_url)}"><error>Failed to process PDF: {escape_xml(str(e))}</error></source>'
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


def fetch_youtube_transcript(url):
    """
    Fetches YouTube transcript using youtube_transcript_api, wrapped in XML.
    """
    def extract_video_id(url):
        pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, url)
        return match.group(1) if match else None

    video_id = extract_video_id(url)
    if not video_id:
        print(f"[bold red]Could not extract YouTube video ID from URL: {url}[/bold red]")
        # Use XML for errors
        return f'<source type="youtube_transcript" url="{escape_xml(url)}">\n<error>Could not extract video ID from URL.</error>\n</source>'

    try:
        print(f"Fetching transcript for YouTube video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        transcript = formatter.format_transcript(transcript_list)
        print("Transcript fetched successfully.")

        # Use XML structure for success
        formatted_text = f'<source type="youtube_transcript" url="{escape_xml(url)}">\n'
        formatted_text += transcript  # Append raw transcript text
        formatted_text += '\n</source>'  # Close source tag
        return formatted_text
    except Exception as e:
        print(f"[bold red]Error fetching YouTube transcript for {url}: {e}[/bold red]")
        # Use XML structure for errors
        return f'<source type="youtube_transcript" url="{escape_xml(url)}">\n<error>{escape_xml(str(e))}</error>\n</source>'


def preprocess_text(input_file, output_file):
    """
    Preprocesses text, optionally removing stopwords if NLTK is enabled.
    Handles potential XML structure if present (intended for compressed output).
    """
    print("Preprocessing text for compression (if enabled)...")
    with open(input_file, "r", encoding="utf-8") as infile:
        input_text = infile.read()

    def process_content(text):
        text = re.sub(r"[\n\r]+", "\n", text)
        text = re.sub(r"[^a-zA-Z0-9\s_.,!?:;@#$%^&*()+\-=[\]{}|\\<>`~'\"/]+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        if ENABLE_COMPRESSION_AND_NLTK and stop_words:
            words = text.split()
            words = [word for word in words if word not in stop_words]
            text = " ".join(words)
        return text

    try:
        # Attempt to parse as XML - this is mainly relevant if the INPUT
        # already had some structure we wanted to preserve during compression
        root = ET.fromstring(input_text)
        for elem in root.iter():
            if elem.text:
                elem.text = process_content(elem.text)
            if elem.tail:
                elem.tail = process_content(elem.tail)
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print("Text preprocessing with XML structure preservation completed.")
    except ET.ParseError:
        # If input is not valid XML (likely our case with raw content), process as plain text
        processed_text = process_content(input_text)
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(processed_text)
        print("Input was not XML. Text preprocessing completed as plain text.")
    except Exception as e:
        print(f"[bold red]Error during text preprocessing: {e}[/bold red]")
        # Fallback: write the original text if preprocessing fails
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(input_text)
        print("[bold yellow]Warning:[/bold yellow] Preprocessing failed, writing original content to compressed file.")


def get_token_count(text, disallowed_special=[], chunk_size=1000):
    """
    Counts tokens using tiktoken, stripping XML tags first.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    # Restore XML tag removal before counting tokens
    # This gives a count of the actual content, not the structural tags
    text_without_tags = re.sub(r'<[^>]+>', '', text)

    # Split the text without tags into smaller chunks for more robust encoding
    chunks = [text_without_tags[i:i+chunk_size] for i in range(0, len(text_without_tags), chunk_size)]
    total_tokens = 0

    for chunk in chunks:
        try:
            tokens = enc.encode(chunk, disallowed_special=disallowed_special)
            total_tokens += len(tokens)
        except Exception as e:
            print(f"[bold yellow]Warning:[/bold yellow] Error encoding chunk for token count: {e}")
            # Estimate token count for problematic chunk (e.g., len/4)
            total_tokens += len(chunk) // 4

    return total_tokens


def is_same_domain(base_url, new_url):
    return urlparse(base_url).netloc == urlparse(new_url).netloc


def is_within_depth(base_url, current_url, max_depth):
    # Simple path depth check
    base_path = urlparse(base_url).path.rstrip('/')
    current_path = urlparse(current_url).path.rstrip('/')
    # Ensure current path starts with base path
    if not current_path.startswith(base_path):
        return False
    base_depth = len(base_path.split('/')) if base_path else 0
    current_depth = len(current_path.split('/')) if current_path else 0
    return (current_depth - base_depth) <= max_depth


def process_web_pdf(url):
    """Downloads and extracts text from a PDF found during web crawl."""
    temp_pdf_path = 'temp_web.pdf'
    try:
        print(f"  Downloading PDF: {url}")
        response = requests.get(url, timeout=30)  # Add timeout
        response.raise_for_status()

        # Basic check for PDF content type
        if 'application/pdf' not in response.headers.get('Content-Type', '').lower():
            print(f"  [bold yellow]Warning:[/bold yellow] URL doesn't report as PDF, skipping: {url}")
            return None  # Or return an error string

        with open(temp_pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)

        print(f"  Extracting text from PDF: {url}")
        text_list = []
        with open(temp_pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page].extract_text()
                if page_text:
                    text_list.append(page_text)
        return ' '.join(text_list)
    except requests.RequestException as e:
        print(f"  [bold red]Error downloading PDF {url}: {e}[/bold red]")
        return f"<error>Failed to download PDF: {escape_xml(str(e))}</error>"
    except Exception as e:
        print(f"  [bold red]Error processing PDF {url}: {e}[/bold red]")
        return f"<error>Failed to process PDF: {escape_xml(str(e))}</error>"
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


def crawl_and_extract_text(base_url, max_depth, include_pdfs, ignore_epubs):
    """
    Crawls a website starting from base_url, extracts text, and wraps in XML.
    """
    visited_urls = set()
    urls_to_visit = [(base_url, 0)]
    processed_urls_content = {}  # Store URL -> content/error
    # Start XML structure
    all_text = [f'<source type="web_crawl" base_url="{escape_xml(base_url)}">']

    print(f"Starting crawl from: {base_url} (Max Depth: {max_depth}, Include PDFs: {include_pdfs})")

    while urls_to_visit:
        current_url, current_depth = urls_to_visit.pop(0)
        # Normalize URL: remove fragment and ensure scheme
        parsed_url = urlparse(current_url)
        clean_url = urlparse(current_url)._replace(fragment="").geturl()
        if not parsed_url.scheme:
            clean_url = "http://" + clean_url  # Default to http if missing

        if clean_url in visited_urls:
            continue

        # Check domain and depth *after* cleaning URL
        if not is_same_domain(base_url, clean_url) or not is_within_depth(base_url, clean_url, max_depth):
            # print(f"Skipping (domain/depth): {clean_url}") # Optional debug
            continue

        if ignore_epubs and clean_url.lower().endswith('.epub'):
            print(f"Skipping (EPUB): {clean_url}")
            visited_urls.add(clean_url)
            continue

        print(f"Processing (Depth {current_depth}): {clean_url}")
        visited_urls.add(clean_url)
        page_content = f'\n<page url="{escape_xml(clean_url)}">'  # Start page tag

        try:
            # Handle PDFs separately
            if clean_url.lower().endswith('.pdf'):
                if include_pdfs:
                    pdf_text = process_web_pdf(clean_url)
                    if pdf_text:  # Append text or error message from process_web_pdf
                        page_content += f'\n{pdf_text}\n'
                    else:  # process_web_pdf returned None (e.g., wrong content type)
                        page_content += '\n<error>Skipped non-PDF content reported at PDF URL.</error>\n'
                else:
                    print(f"  Skipping PDF (include_pdfs=False): {clean_url}")
                    page_content += '\n<skipped>PDF ignored by configuration</skipped>\n'

            # Handle HTML pages
            else:
                # Add timeout to requests
                response = requests.get(clean_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
                response.raise_for_status()

                # Basic check for HTML content type
                if 'text/html' not in response.headers.get('Content-Type', '').lower():
                    print(f"  [bold yellow]Warning:[/bold yellow] Skipping non-HTML page: {clean_url} (Content-Type: {response.headers.get('Content-Type')})")
                    page_content += f'\n<skipped>Non-HTML content type: {escape_xml(response.headers.get("Content-Type", "N/A"))}</skipped>\n'
                else:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Remove scripts, styles, etc.
                    for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]', 'nav', 'footer', 'aside']):  # Added common noise tags
                        element.decompose()
                    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
                    for comment in comments:
                        comment.extract()
                    # Get text, try to preserve some structure with newlines
                    text = soup.get_text(separator='\n', strip=True)
                    page_content += f'\n{text}\n'  # Append raw extracted text

                    # Find links for the next level if depth allows
                    if current_depth < max_depth:
                        for link in soup.find_all('a', href=True):
                            try:
                                new_url_raw = link['href']
                                if new_url_raw and not new_url_raw.startswith(('mailto:', 'javascript:', '#')):
                                    new_url = urljoin(clean_url, new_url_raw)
                                    parsed_new = urlparse(new_url)
                                    # Add scheme if missing for domain/depth checks
                                    if not parsed_new.scheme:
                                        new_url = parsed_new._replace(scheme=urlparse(clean_url).scheme).geturl()

                                    new_clean_url = urlparse(new_url)._replace(fragment="").geturl()

                                    if new_clean_url not in visited_urls:
                                        # Check domain/depth *before* adding to queue
                                        if is_same_domain(base_url, new_clean_url) and is_within_depth(base_url, new_clean_url, max_depth):
                                            if not (ignore_epubs and new_clean_url.lower().endswith('.epub')):
                                                # Add only if valid and not already visited
                                                if (new_clean_url, current_depth + 1) not in urls_to_visit:
                                                    urls_to_visit.append((new_clean_url, current_depth + 1))
                            except Exception as link_err:  # Catch errors parsing individual links
                                print(f"  [bold yellow]Warning:[/bold yellow] Error parsing link '{link.get('href')}': {link_err}")

        except requests.exceptions.Timeout:
            print(f"[bold red]Timeout retrieving {clean_url}[/bold red]")
            page_content += f'\n<error>Timeout during request</error>\n'
        except requests.RequestException as e:
            print(f"[bold red]Failed to retrieve {clean_url}: {e}[/bold red]")
            page_content += f'\n<error>Failed to retrieve URL: {escape_xml(str(e))}</error>\n'
        except Exception as e:  # Catch other errors like BeautifulSoup issues
            print(f"[bold red]Error processing page {clean_url}: {e}[/bold red]")
            page_content += f'\n<error>Error processing page: {escape_xml(str(e))}</error>\n'

        page_content += '</page>'  # Close page tag
        all_text.append(page_content)
        processed_urls_content[clean_url] = page_content  # Store for processed list

    all_text.append('\n</source>')  # Close source tag
    print("Web crawl finished.")
    formatted_content = '\n'.join(all_text)

    return {
        'content': formatted_content,
        'processed_urls': list(processed_urls_content.keys())  # Return URLs we attempted to process
    }


def process_doi_or_pmid(identifier):
    """
    Attempts to fetch a paper PDF via Sci-Hub using DOI or PMID, wrapped in XML.
    Note: Sci-Hub access can be unreliable.
    """
    # Use a more reliable Sci-Hub domain if known, otherwise fallback
    sci_hub_domains = ['https://sci-hub.se/', 'https://sci-hub.st/', 'https://sci-hub.ru/']  # Add more mirrors if needed
    pdf_filename = f"temp_{identifier.replace('/', '-')}.pdf"
    pdf_text = None

    for base_url in sci_hub_domains:
        print(f"Attempting Sci-Hub domain: {base_url} for identifier: {identifier}")
        headers = {  # Headers might help avoid blocks
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }
        payload = {'request': identifier}

        try:
            # Initial request to Sci-Hub page
            response = requests.post(base_url, headers=headers, data=payload, timeout=60)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the PDF link/embed (Sci-Hub structure varies)
            pdf_url = None
            # Try common patterns: iframe#pdf, button onclick location.href, direct links
            pdf_frame = soup.find('iframe', id='pdf')
            if pdf_frame and pdf_frame.get('src'):
                pdf_url = urljoin(base_url, pdf_frame['src'])
            else:
                # Look for buttons or links directing to the PDF
                pdf_button = soup.find('button', onclick=lambda x: x and 'location.href=' in x)
                if pdf_button:
                    match = re.search(r"location\.href='(//.*?)'", pdf_button['onclick'])
                    if match:
                        # Need to add scheme if missing (often //...)
                        pdf_url_part = match.group(1)
                        if pdf_url_part.startswith("//"):
                            pdf_url = "https:" + pdf_url_part
                        else:
                            pdf_url = urljoin(base_url, pdf_url_part)

            if not pdf_url:
                print(f"  Could not find PDF link on page from {base_url}")
                continue  # Try next domain

            print(f"  Found potential PDF URL: {pdf_url}")
            # Ensure URL has scheme for requests
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            elif not pdf_url.startswith("http"):
                pdf_url = urljoin(base_url, pdf_url)

            print(f"  Downloading PDF from: {pdf_url}")
            # Download the PDF file
            pdf_response = requests.get(pdf_url, headers=headers, timeout=120)  # Longer timeout for PDF download
            pdf_response.raise_for_status()

            # Check content type again
            if 'application/pdf' not in pdf_response.headers.get('Content-Type', '').lower():
                print(f"  [bold yellow]Warning:[/bold yellow] Downloaded content is not PDF from {pdf_url}, trying next domain.")
                continue

            with open(pdf_filename, 'wb') as f:
                f.write(pdf_response.content)

            print("  Extracting text from PDF...")
            with open(pdf_filename, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                text_list = []
                for page in range(len(pdf_reader.pages)):
                    page_text = pdf_reader.pages[page].extract_text()
                    if page_text:
                        text_list.append(page_text)
                pdf_text = " ".join(text_list)

            print(f"Identifier {identifier} processed successfully via {base_url}.")
            break  # Success, exit the loop

        except requests.exceptions.Timeout:
            print(f"  Timeout connecting to {base_url} or downloading PDF.")
            continue  # Try next domain
        except requests.RequestException as e:
            print(f"  Error with {base_url}: {e}")
            continue  # Try next domain
        except Exception as e:  # Catch other errors (PDF parsing, etc.)
            print(f"  Error processing identifier {identifier} with {base_url}: {e}")
            continue  # Try next domain
        finally:
            # Clean up temp file even if loop continues
            if os.path.exists(pdf_filename):
                os.remove(pdf_filename)

    # After trying all domains
    if pdf_text is not None:
        # Use XML structure for success
        formatted_text = f'<source type="sci-hub" identifier="{escape_xml(identifier)}">\n'
        formatted_text += pdf_text  # Append raw extracted text
        formatted_text += '\n</source>'
        return formatted_text
    else:
        print(f"[bold red]Failed to process identifier {identifier} using all Sci-Hub domains tried.[/bold red]")
        # Use XML structure for error
        error_text = f'<source type="sci-hub" identifier="{escape_xml(identifier)}">\n'
        error_text += f'<error>Could not retrieve or process PDF via Sci-Hub.</error>\n'
        error_text += '</source>'
        return error_text


def process_github_pull_request(pull_request_url):
    """
    Processes a GitHub Pull Request, including details, diff, comments, and associated repo content, wrapped in XML.
    """
    if TOKEN == 'default_token_here':
        print("[bold red]Error:[/bold red] GitHub Token not set. Cannot process GitHub Pull Request.")
        return f'<source type="github_pull_request" url="{escape_xml(pull_request_url)}"><error>GitHub Token not configured.</error></source>'

    url_parts = pull_request_url.split("/")
    if len(url_parts) < 7 or url_parts[-2] != 'pull':
        print(f"[bold red]Invalid GitHub Pull Request URL: {pull_request_url}[/bold red]")
        return f'<source type="github_pull_request" url="{escape_xml(pull_request_url)}"><error>Invalid URL format.</error></source>'

    repo_owner = url_parts[3]
    repo_name = url_parts[4]
    pull_request_number = url_parts[-1]

    api_base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pull_request_number}"
    repo_url_for_content = f"https://github.com/{repo_owner}/{repo_name}"  # Base repo URL

    try:
        print(f"Fetching PR data for: {pull_request_url}")
        response = requests.get(api_base_url, headers=headers)
        response.raise_for_status()
        pull_request_data = response.json()

        # Start XML structure
        formatted_text_list = [f'<source type="github_pull_request" url="{escape_xml(pull_request_url)}">']
        formatted_text_list.append(f'<title>{escape_xml(pull_request_data.get("title", "N/A"))}</title>')  # Use .get for safety
        formatted_text_list.append('<description>')
        formatted_text_list.append(pull_request_data.get('body', "") or "")  # Append raw body, handle None
        formatted_text_list.append('</description>')
        details = (
            f"User: {pull_request_data.get('user', {}).get('login', 'N/A')}, "
            f"State: {pull_request_data.get('state', 'N/A')}, "
            f"Commits: {pull_request_data.get('commits', 'N/A')}, "
            f"Base: {pull_request_data.get('base', {}).get('label', 'N/A')}, "
            f"Head: {pull_request_data.get('head', {}).get('label', 'N/A')}"
        )
        formatted_text_list.append(f'<details>{escape_xml(details)}</details>')

        # Fetch and add the diff
        diff_url = pull_request_data.get("diff_url")
        if diff_url:
            print("Fetching PR diff...")
            diff_response = requests.get(diff_url, headers=headers)
            diff_response.raise_for_status()
            pull_request_diff = diff_response.text
            formatted_text_list.append('\n<diff>')
            formatted_text_list.append(pull_request_diff)  # Append raw diff
            formatted_text_list.append('</diff>')
        else:
            formatted_text_list.append('\n<diff><error>Could not retrieve diff URL.</error></diff>')

        # Fetch and add comments (PR comments + review comments)
        all_comments_data = []
        comments_url = pull_request_data.get("comments_url")
        review_comments_url = pull_request_data.get("review_comments_url")

        if comments_url:
            print("Fetching PR comments...")
            comments_response = requests.get(comments_url, headers=headers)
            if comments_response.ok:
                all_comments_data.extend(comments_response.json())
            else:
                print(f"[bold yellow]Warning:[/bold yellow] Could not fetch PR comments: {comments_response.status_code}")

        if review_comments_url:
            print("Fetching PR review comments...")
            review_comments_response = requests.get(review_comments_url, headers=headers)
            if review_comments_response.ok:
                all_comments_data.extend(review_comments_response.json())
            else:
                print(f"[bold yellow]Warning:[/bold yellow] Could not fetch review comments: {review_comments_response.status_code}")

        if all_comments_data:
            formatted_text_list.append('\n<comments>')
            # Optional: Sort comments by creation date or position
            all_comments_data.sort(key=lambda c: c.get("created_at", ""))
            for comment in all_comments_data:
                author = comment.get('user', {}).get('login', 'N/A')
                body = comment.get('body', '') or ""  # Handle None
                # Add context if available (e.g., path, line for review comments)
                path = comment.get('path')
                line = comment.get('line') or comment.get('original_line')
                context = f' path="{escape_xml(path)}"' if path else ''
                context += f' line="{line}"' if line else ''
                formatted_text_list.append(f'<comment author="{escape_xml(author)}"{context}>')
                formatted_text_list.append(body)  # Append raw comment body
                formatted_text_list.append('</comment>')
            formatted_text_list.append('</comments>')

        # Add repository content (will include its own <source> tag)
        print(f"Fetching associated repository content from: {repo_url_for_content}")
        # Use the base branch if available, otherwise default branch content
        base_branch_ref = pull_request_data.get('base', {}).get('ref')
        repo_url_with_ref = f"{repo_url_for_content}/tree/{base_branch_ref}" if base_branch_ref else repo_url_for_content
        repo_content = process_github_repo(repo_url_with_ref)  # process_github_repo returns XML string

        formatted_text_list.append('\n<!-- Associated Repository Content -->')  # XML Comment
        formatted_text_list.append(repo_content)  # Append the XML output directly

        formatted_text_list.append('\n</source>')  # Close main PR source tag

        print(f"Pull request {pull_request_number} and repository content processed successfully.")
        return "\n".join(formatted_text_list)

    except requests.RequestException as e:
        print(f"[bold red]Error fetching GitHub PR data for {pull_request_url}: {e}[/bold red]")
        return f'<source type="github_pull_request" url="{escape_xml(pull_request_url)}"><error>Failed to fetch PR data: {escape_xml(str(e))}</error></source>'
    except Exception as e:  # Catch other potential errors
        print(f"[bold red]Unexpected error processing GitHub PR {pull_request_url}: {e}[/bold red]")
        return f'<source type="github_pull_request" url="{escape_xml(pull_request_url)}"><error>Unexpected error: {escape_xml(str(e))}</error></source>'


def process_github_issue(issue_url):
    """
    Processes a GitHub Issue, including details, comments, and associated repo content, wrapped in XML.
    """
    if TOKEN == 'default_token_here':
        print("[bold red]Error:[/bold red] GitHub Token not set. Cannot process GitHub Issue.")
        return f'<source type="github_issue" url="{escape_xml(issue_url)}"><error>GitHub Token not configured.</error></source>'

    url_parts = issue_url.split("/")
    if len(url_parts) < 7 or url_parts[-2] != 'issues':
        print(f"[bold red]Invalid GitHub Issue URL: {issue_url}[/bold red]")
        return f'<source type="github_issue" url="{escape_xml(issue_url)}"><error>Invalid URL format.</error></source>'

    repo_owner = url_parts[3]
    repo_name = url_parts[4]
    issue_number = url_parts[-1]

    api_base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}"
    repo_url_for_content = f"https://github.com/{repo_owner}/{repo_name}"

    try:
        print(f"Fetching issue data for: {issue_url}")
        response = requests.get(api_base_url, headers=headers)
        response.raise_for_status()
        issue_data = response.json()

        # Start XML structure
        formatted_text_list = [f'<source type="github_issue" url="{escape_xml(issue_url)}">']
        formatted_text_list.append(f'<title>{escape_xml(issue_data.get("title", "N/A"))}</title>')
        formatted_text_list.append('<description>')
        formatted_text_list.append(issue_data.get('body', "") or "")  # Append raw body, handle None
        formatted_text_list.append('</description>')
        details = (
            f"User: {issue_data.get('user', {}).get('login', 'N/A')}, "
            f"State: {issue_data.get('state', 'N/A')}, "
            f"Number: {issue_data.get('number', 'N/A')}"
        )
        formatted_text_list.append(f'<details>{escape_xml(details)}</details>')

        # Fetch and add comments
        comments_data = []
        comments_url = issue_data.get("comments_url")
        if comments_url:
            print("Fetching issue comments...")
            comments_response = requests.get(comments_url, headers=headers)
            if comments_response.ok:
                comments_data = comments_response.json()
            else:
                print(f"[bold yellow]Warning:[/bold yellow] Could not fetch issue comments: {comments_response.status_code}")

        if comments_data:
            formatted_text_list.append('\n<comments>')
            # Optional: Sort comments by creation date
            comments_data.sort(key=lambda c: c.get("created_at", ""))
            for comment in comments_data:
                author = comment.get('user', {}).get('login', 'N/A')
                body = comment.get('body', '') or ""  # Handle None
                formatted_text_list.append(f'<comment author="{escape_xml(author)}">')
                formatted_text_list.append(body)  # Append raw comment body
                formatted_text_list.append('</comment>')
            formatted_text_list.append('</comments>')

        # Add repository content (will include its own <source> tag)
        print(f"Fetching associated repository content from: {repo_url_for_content}")
        # Fetch default branch content for issues
        repo_content = process_github_repo(repo_url_for_content)  # process_github_repo returns XML string

        formatted_text_list.append('\n<!-- Associated Repository Content -->')  # XML Comment
        formatted_text_list.append(repo_content)  # Append the XML output directly

        formatted_text_list.append('\n</source>')  # Close main issue source tag

        print(f"Issue {issue_number} and repository content processed successfully.")
        return "\n".join(formatted_text_list)

    except requests.RequestException as e:
        print(f"[bold red]Error fetching GitHub issue data for {issue_url}: {e}[/bold red]")
        return f'<source type="github_issue" url="{escape_xml(issue_url)}"><error>Failed to fetch issue data: {escape_xml(str(e))}</error></source>'
    except Exception as e:  # Catch other potential errors
        print(f"[bold red]Unexpected error processing GitHub issue {issue_url}: {e}[/bold red]")
        return f'<source type="github_issue" url="{escape_xml(issue_url)}"><error>Unexpected error: {escape_xml(str(e))}</error></source>'


def is_excluded_file(filename):
    """Check if a file should be excluded based on patterns."""
    excluded_patterns = [
        '.pb.go', '_grpc.pb.go', 'mock_', '/generated/', '/mocks/', '.gen.', '_generated.',
        # Add common compiled/generated file patterns
        '.min.js', '.min.css', '.dll', '.o', '.so', '.a', '.class', '.pyc',
        # Common dependency/build directories already handled by EXCLUDED_DIRS, but can add file patterns too
        'package-lock.json', 'yarn.lock', 'go.sum',
    ]
    # Check basename and full path for flexibility
    basename = os.path.basename(filename)
    return any(pattern in filename for pattern in excluded_patterns) or \
        any(basename.startswith(pattern) for pattern in ['mock_']) or \
        any(basename.endswith(pattern) for pattern in ['.pyc', '.pb.go', '_grpc.pb.go'])


def is_allowed_filetype(filename):
    """Check if a file should be processed based on its extension and exclusion patterns."""
    if is_excluded_file(filename):
        return False

    # Prioritize text-based formats commonly used in code/docs
    allowed_extensions = [
        # Code
        '.py', '.go', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.js', '.ts', '.jsx', '.tsx',
        '.rb', '.php', '.swift', '.kt', '.scala', '.rs', '.lua', '.pl', '.sh', '.bash', '.zsh',
        '.ps1', '.sql', '.groovy', '.dart',
        # Config/Data
        '.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.cfg', '.conf', '.properties',
        '.csv', '.tsv', '.proto', '.graphql', '.tf', '.tfvars', '.hcl',
        # Markup/Docs
        '.md', '.txt', '.rst', '.tex', '.html', '.htm', '.css', '.scss', '.less',
        # Notebooks
        '.ipynb',
        # Other useful text
        '.dockerfile', 'Dockerfile', '.gitignore', '.gitattributes', 'Makefile', '.env',
        '.cjs', '.localhost', '.example',  # From original list
        'go.mod',  # Go module file
    ]
    # Check for exact filename matches first (like Dockerfile)
    basename = os.path.basename(filename)
    if basename in ['Dockerfile', 'Makefile', '.gitignore', '.gitattributes', 'go.mod']:
        return True
    # Then check extensions
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def combine_xml_outputs(outputs):
    """
    Combines multiple XML outputs into one cohesive XML document.
    Each input should be a complete XML document with <source> tags.
    """
    if not outputs:
        return None

    # If only one output, return it as is
    if len(outputs) == 1:
        return outputs[0]

    # Create a wrapper for multiple sources
    combined = ['<combined_sources>']

    # Add each source
    for output in outputs:
        # Remove any XML declaration if present (rare but possible)
        output = re.sub(r'<\?xml[^>]+\?>', '', output).strip()
        combined.append(output)

    # Close the wrapper
    combined.append('</combined_sources>')

    return '\n'.join(combined)


def process_input(input_path, progress=None, task=None):
    """
    Process a single input path and return the XML output.
    Extracted from main() for reuse with multiple inputs.
    """
    console = Console()
    urls_list_file = "processed_urls.txt"

    try:
        if task:
            progress.update(task, description=f"[bright_blue]Processing {input_path}...")

        console.print(f"\n[bold bright_green]Processing:[/bold bright_green] [bold bright_yellow]{input_path}[/bold bright_yellow]\n")

        # Input type detection logic
        if "github.com" in input_path:
            if "/pull/" in input_path:
                result = process_github_pull_request(input_path)
            elif "/issues/" in input_path:
                result = process_github_issue(input_path)
            else:  # Assume repository URL
                result = process_github_repo(input_path)
        elif urlparse(input_path).scheme in ["http", "https"]:
            if "youtube.com" in input_path or "youtu.be" in input_path:
                result = fetch_youtube_transcript(input_path)
            elif "arxiv.org/abs/" in input_path:
                result = process_arxiv_pdf(input_path)
            elif input_path.lower().endswith(('.pdf')):  # Direct PDF link
                # Simplified: wrap direct PDF processing if needed, or treat as web crawl
                print("[bold yellow]Direct PDF URL detected - treating as single-page crawl.[/bold yellow]")
                crawl_result = crawl_and_extract_text(input_path, max_depth=0, include_pdfs=True, ignore_epubs=True)
                result = crawl_result['content']
                if crawl_result['processed_urls']:
                    with open(urls_list_file, 'w', encoding='utf-8') as urls_file:
                        urls_file.write('\n'.join(crawl_result['processed_urls']))
            else:  # Assume general web URL for crawling
                crawl_result = crawl_and_extract_text(input_path, max_depth=1, include_pdfs=True, ignore_epubs=True)  # Default max_depth=1
                result = crawl_result['content']
                if crawl_result['processed_urls']:
                    with open(urls_list_file, 'w', encoding='utf-8') as urls_file:
                        urls_file.write('\n'.join(crawl_result['processed_urls']))
        # Basic check for DOI (starts with 10.) or PMID (all digits)
        elif (input_path.startswith("10.") and "/" in input_path) or input_path.isdigit():
            result = process_doi_or_pmid(input_path)
        elif os.path.isdir(input_path):  # Check if it's a local directory
            result = process_local_folder(input_path)
        elif os.path.isfile(input_path):  # Handle single local file
            print(f"Processing single local file: {input_path}")
            relative_path = os.path.basename(input_path)
            file_content = safe_file_read(input_path)
            # Wrap single file in basic source/file XML
            result = (f'<source type="local_file" path="{escape_xml(input_path)}">\n'
                      f'<file path="{escape_xml(relative_path)}">\n'
                      f'{file_content}\n'
                      f'</file>\n'
                      f'</source>')
        else:  # Input not recognized
            raise ValueError(f"Input path or URL type not recognized: {input_path}")

        return result

    except Exception as e:
        console.print(f"\n[bold red]Error processing {input_path}:[/bold red] {str(e)}")
        # Return an error-wrapped source instead of raising
        return f'<source type="error" path="{escape_xml(input_path)}">\n<e>Failed to process: {escape_xml(str(e))}</e>\n</source>'


def main():
    console = Console()

    # Updated intro text to reflect XML output
    intro_text = Text("\nProcesses Inputs and Wraps Content in XML:\n", style="dodger_blue1")
    input_types = [
        ("• Local folder path", "bright_white"),
        ("• GitHub repository URL", "bright_white"),
        ("• GitHub pull request URL (PR + Repo)", "bright_white"),
        ("• GitHub issue URL (Issue + Repo)", "bright_white"),
        ("• Documentation URL (Web Crawl)", "bright_white"),
        ("• YouTube video URL (Transcript)", "bright_white"),
        ("• ArXiv Paper URL (PDF Text)", "bright_white"),
        ("• DOI or PMID (via Sci-Hub, best effort)", "bright_white"),
    ]

    for input_type, color in input_types:
        intro_text.append(f"\n{input_type}", style=color)
    intro_text.append("\n\nOutput is saved to file and copied to clipboard.", style="dim")
    intro_text.append("\nContent within XML tags remains unescaped for readability.", style="dim")
    intro_text.append("\nMultiple inputs can be provided as command line arguments.", style="bright_green")

    intro_panel = Panel(
        intro_text,
        expand=False,
        border_style="bold",
        title="[bright_white]onefilellm - Content Aggregator[/bright_white]",
        title_align="center",
        padding=(1, 1),
    )
    console.print(intro_panel)

    # Determine input paths - either from command line args or user prompt
    input_paths = []
    if len(sys.argv) > 1:
        input_paths = sys.argv[1:]
    else:
        input_path = Prompt.ask("\n[bold dodger_blue1]Enter the path or URL[/bold dodger_blue1]", console=console)
        input_paths = [input_path]

    # Define output filenames
    output_file = "output.xml"  # Changed extension to reflect content
    processed_file = "compressed_output.txt"  # Keep as txt for compressed

    # List to collect individual outputs
    outputs = []

    with Progress(
        TextColumn("[bold bright_blue]{task.description}"),
        BarColumn(bar_width=None),
        TimeRemainingColumn(),
        console=console,
        transient=True  # Clear progress on exit
    ) as progress:

        task = progress.add_task("[bright_blue]Processing...", total=None)  # Indeterminate task

        try:
            # Process each input path
            for input_path in input_paths:
                result = process_input(input_path, progress, task)
                if result:
                    outputs.append(result)
                    console.print(f"[green]Successfully processed: {input_path}[/green]")
                else:
                    console.print(f"[yellow]No output generated for: {input_path}[/yellow]")

            # Combine all outputs into one final output
            if not outputs:
                raise RuntimeError("No inputs were successfully processed.")

            final_output = combine_xml_outputs(outputs)

            # --- Output Generation ---
            if final_output is None:
                raise RuntimeError("Processing failed to produce any output.")

            # Write the main (uncompressed) XML output
            print(f"\nWriting output to {output_file}...")
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(final_output)
            print("Output written successfully.")

            # Get token count for the main output (strips XML tags)
            uncompressed_token_count = get_token_count(final_output)
            console.print(f"\n[bright_green]Content Token Count (approx):[/bright_green] [bold bright_cyan]{uncompressed_token_count}[/bold bright_cyan]")

            # --- Conditional Compression Block ---
            if ENABLE_COMPRESSION_AND_NLTK:
                console.print("\n[bright_magenta]Compression and NLTK processing enabled.[/bright_magenta]")
                print(f"Preprocessing text and writing compressed output to {processed_file}...")
                # Process the text (input is the XML file, output is compressed txt)
                preprocess_text(output_file, processed_file)
                print("Compressed file written.")

                # Get token count for the compressed file (should be plain text)
                compressed_text = safe_file_read(processed_file)
                compressed_token_count = get_token_count(compressed_text)  # Pass compressed text directly
                console.print(f"[bright_green]Compressed Token Count (approx):[/bright_green] [bold bright_cyan]{compressed_token_count}[/bold bright_cyan]")
                console.print(f"\n[bold bright_blue]{output_file}[/bold bright_blue] (main XML) and [bold bright_yellow]{processed_file}[/bold bright_yellow] (compressed text) created.")
            else:
                console.print(f"\n[bold bright_blue]{output_file}[/bold bright_blue] (main XML) has been created.")
            # --- End Conditional Compression Block ---

            # Copy the main XML output to clipboard
            try:
                pyperclip.copy(final_output)
                console.print(f"\n[bright_white]The contents of [bold bright_blue]{output_file}[/bold bright_blue] have been copied to the clipboard.[/bright_white]")
            except Exception as clip_err:  # Catch potential pyperclip errors
                console.print(f"\n[bold yellow]Warning:[/bold yellow] Could not copy to clipboard: {clip_err}")

        except Exception as e:
            console.print(f"\n[bold red]An error occurred during processing:[/bold red]")
            console.print_exception(show_locals=False)  # Print traceback
            # Optionally write the partial output if it exists
            if outputs:
                try:
                    error_filename = "error_output.xml"
                    partial_output = combine_xml_outputs(outputs)
                    with open(error_filename, "w", encoding="utf-8") as err_file:
                        err_file.write(partial_output)
                    console.print(f"[yellow]Partial output written to {error_filename}[/yellow]")
                except Exception as write_err:
                    console.print(f"[red]Could not write partial output: {write_err}[/red]")

        finally:
            progress.stop_task(task)
            progress.refresh()  # Ensure progress bar clears


if __name__ == "__main__":
    main()
