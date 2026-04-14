import argparse
import base64
import warnings
import os
import logging
import requests
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config import MODELS, API_KEYS

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def get_chat_client_kwargs(provider, api_keys):
    """Return provider-specific kwargs for ChatOpenAI."""
    if provider == 'openai':
        return {
            "openai_api_key": api_keys['openai'],
        }

    if provider == 'gemini':
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is required when --provider gemini is used.")
        return {
            "openai_api_key": gemini_key,
            "openai_api_base": GEMINI_BASE_URL,
        }

    raise ValueError(f"Unsupported provider: {provider}")


def configure_chat_client(model_name, provider, api_keys, temperature, max_tokens=None,
                          seed=None):
    """Create a ChatOpenAI client for either OpenAI or Gemini."""
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    kwargs.update(get_chat_client_kwargs(provider, api_keys))

    if seed is not None and provider == 'openai':
        kwargs["model_kwargs"] = {"seed": seed}

    return ChatOpenAI(**kwargs)


def initialize_clients(modeLs, api_keys, provider='openai', chat_model=None,
                       vision_model=None):
    llm = configure_chat_client(
        chat_model or modeLs['llm'],
        provider,
        api_keys,
        temperature=0.3,
    )
    llm_vision = configure_chat_client(
        vision_model or modeLs['llm_vision'],
        provider,
        api_keys,
        temperature=0.1,
        max_tokens=1000,
    )
    embedding = OpenAIEmbeddings(
        model=modeLs['embedding'],
        openai_api_key=api_keys['openai']
    )
    return llm, llm_vision, embedding

def load_data(file_path):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    return data

def setup_vector_database(data, embedding):
    attempts = 0
    max_attempts = 5  # You can set this to the number of retry attempts you find appropriate
    while attempts < max_attempts:
        try:
            vectordb = Chroma.from_documents(documents=data, embedding=embedding, collection_name="openai_embed")
            return vectordb  # Successful creation, return the database
        except Exception as e:
            logging.error(f"Failed to setup vector database: {e}")
            attempts += 1
            time.sleep(180)  # Wait for 30 seconds before retrying
            
    # If the loop completes without returning, log final failure and optionally raise an exception
    logging.critical("Failed to initialize vector database after multiple attempts.")
    raise Exception("Failed to initialize vector database after multiple attempts.")

def load_image_as_data_uri(path_or_url):
    """Convert a local file path to a base64 data URI, or return URL as-is."""
    if os.path.isfile(path_or_url):
        ext = os.path.splitext(path_or_url)[1].lower()
        image_type = 'jpeg' if ext in ('.jpg', '.jpeg') else 'png'
        with open(path_or_url, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return f"data:image/{image_type};base64,{encoded}"
    return path_or_url


# Helper functions for message generation and checkpointing
def get_messages_from_url(url_str):
    """Generate a sequence of messages for a URL or local path containing an image."""
    return [
        SystemMessage(
            content="You are an expert at analyzing images with computer vision. "
                    "I will present you with a picture of food, which might be placed on a plate, inside a spoon, "
                    "or contained within different vessels. Your job is to accurately identify the food depicted in the image."
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": """Could you name the food shown in the image? If feasible, specify its variety
                 based on ingredients and preparation methods. Keep your response concise.

If you are uncertain about any aspect of your identification, naturally incorporate
qualifying words such as 'likely', 'appears to be', 'possibly', 'seems like', or 'probably'
into your description. For example: 'appears to be grilled chicken' or 'likely a Caesar salad'.

If you cannot analyze the image at all, respond with 'I can't help to analyze this image.'
and provide the reason on a new line."""},
                {"type": "image_url", "image_url": {"url": url_str}}
            ]
        )
    ]

def setup_retrieval_prompt():
    RETRIEVE_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given food descriptions to retrieve relevant food information from a vector 
        database. These decriptions should docus purely on the qualitative asepcts of the food, including its broader category, specific flavor profiles, ingredients, and methods of preparation. 
        By generating multiple perspectives on the food description, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Please provide these alternate food descriptions, adding the original description as the sixth entry without mentioning that it is the original description. 
        Ensure each of the six descriptions is separated by a newline.
        Original food description: {question}""",
    )
    return RETRIEVE_PROMPT

def configure_retrievers(llm, vectordb, prompt):
    # llm_chain_retriever = LLMChain(llm=llm, prompt=prompt)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), 
        llm=llm, 
        prompt=prompt
    )
    return retriever_from_llm

def setup_code_prompt_chain(retriever_from_llm, llm):
    template = """Could you identify and provide the eight-digit food code corresponding to the given food description: {question},
    considering only the context provided: {context}?

    Please provide only the eight-digit codes without any extra information. If an exact match for the food description is not available, please identify the closest equivalent.
    Should there be no relevant food codes based on the context provided, simply reply with 'No appropriate food codes found from the context information.
    """
    CODE_PROMPT = ChatPromptTemplate.from_template(template)

    food_code_chain = (
        {"context": retriever_from_llm, "question": RunnablePassthrough()}
        | CODE_PROMPT
        | llm
        | StrOutputParser()
    )
    return food_code_chain

def is_integer(s):
    """Check if a string can be converted to an integer."""
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_retrieval_scores_for_description(description, vectordb, llm, retrieval_prompt, k=20):
    """
    Compute retrieval confidence scores for a food description.

    Generates query variations (like MultiQueryRetriever) and runs
    similarity_search_with_score on each, then aggregates scores.

    Args:
        description: Food description text
        vectordb: ChromaDB vector store
        llm: LLM for generating query variations
        retrieval_prompt: PromptTemplate for generating variations
        k: Number of results per query

    Returns:
        dict with retrieval_score_avg, retrieval_score_top1, retrieval_score_spread
    """
    default_scores = {
        'retrieval_score_avg': np.nan,
        'retrieval_score_top1': np.nan,
        'retrieval_score_spread': np.nan
    }

    try:
        # Generate query variations using the retrieval prompt
        variation_response = llm.invoke(
            retrieval_prompt.format(question=description)
        )
        variations = [v.strip() for v in variation_response.content.split('\n') if v.strip()]

        # Add original description if not already included
        if description not in variations:
            variations.append(description)

        # Collect all scores across all query variations
        all_scores = []
        for query in variations:
            try:
                results_with_scores = vectordb.similarity_search_with_score(query, k=k)
                for doc, score in results_with_scores:
                    all_scores.append(score)
            except Exception as e:
                logging.warning(f"Similarity search failed for query variation: {e}")
                continue

        if not all_scores:
            return default_scores

        all_scores = np.array(all_scores)
        return {
            'retrieval_score_avg': float(np.mean(all_scores)),
            'retrieval_score_top1': float(np.min(all_scores)),  # Lower = better in L2
            'retrieval_score_spread': float(np.std(all_scores))
        }

    except Exception as e:
        logging.warning(f"Failed to compute retrieval scores: {e}")
        return default_scores


def process_image_url(index, url, llm_vision, df, food_code_chain, results_path,
                      vectordb=None, llm=None, retrieval_prompt=None):
    """Process a single image (local path or URL) and update the DataFrame with food descriptions and codes."""
    logging.info(f"Processing image at index {index}: {url}")
    try:
        # Check accessibility: local file existence or remote URL
        if os.path.isfile(url):
            image_uri = load_image_as_data_uri(url)
        else:
            req_url = requests.head(url, timeout=5)
            if req_url.status_code != 200:
                raise Exception("Image URL is not accessible")
            image_uri = url

        IMAGE_PROMPT = get_messages_from_url(image_uri)
        attempts = 0  # Counter for retry attempts
        max_attempts = 5 # Set the maximum number of retry attempts
        
        while True:
            try:
                image_response = llm_vision.invoke(IMAGE_PROMPT)
                food_description = image_response.content
                food_description_splits = food_description.split('\n')
                if food_description_splits[0] == "I can't help to analyze this image.":
                    raise Exception(food_description_splits[-1])
                break
            except Exception as e:
                if '429' in str(e) or 'rate limit' in str(e).lower():  # Check if the exception is due to rate limiting
                    if attempts < max_attempts - 1:
                        sleep_time = min(2 ** attempts * 30, 300)  # Exponential backoff with a maximum wait time
                        logging.warning(f"Rate limit exceeded. Retrying in {sleep_time} seconds.")
                        attempts += 1
                        time.sleep(sleep_time)
                        continue  # Continue to retry after sleeping
                    else:
                        logging.error("Maximum retry attempts reached. Failing with error.")
                        raise  # Exception to indicate max attempts have been reached
                else:
                    logging.error(f"Failed to invoke llm_vision at index {index}: {e}")
                    raise  # Re-raise the exception if it's not related to rate limits
                    
        food_code_response = food_code_chain.invoke({"question": food_description, "url_str": url})
        if food_code_response == "No appropriate food codes found from the context information.":
            raise Exception(food_code_response)

        df.loc[index, 'GPTFoodDescription'] = food_description
        df.loc[index, 'GPTFoodCode'] = str(food_code_response.split('\n'))

        # Compute retrieval scores if vectordb is provided
        if vectordb is not None and llm is not None and retrieval_prompt is not None:
            scores = get_retrieval_scores_for_description(
                food_description, vectordb, llm, retrieval_prompt
            )
            for key, value in scores.items():
                df.loc[index, key] = value

    except Exception as e:
        df.loc[index, 'GPTFoodDescription'] = str(e)
        df.loc[index, 'GPTFoodCode'] = np.nan
        # logging.error(f"Failed to process URL at index {index}: {e}")
    finally:
        df.to_csv(results_path, index=False)
        logging.info(f"Data saved to {results_path}")

def load_checkpoint(checkpoint_path):
    """Load the last processed indices from a checkpoint file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as file:
            content = file.read()
            if content:
                num, index = content.split(',')
                return int(num), int(index)
    return 0, 0

def save_checkpoint(num, index, checkpoint_path):
    """Save the current indices to a checkpoint file."""
    with open(checkpoint_path, 'w') as file:
        file.write(f"{num},{index}")
    logging.info(f"Checkpoint saved to {checkpoint_path} at num {num}, index {index}")

def process_image_urls(results_path, checkpoint_path, llm_vision, food_code_chain,
                       vectordb=None, llm=None, retrieval_prompt=None):
    """Process image URLs with checkpointing that includes iterations and index."""
    num_iterations = 5
    df_url = pd.read_csv(results_path)
    last_num, last_index = load_checkpoint(checkpoint_path)

    # Initialize output columns if not present (first run)
    for col in ['GPTFoodDescription', 'GPTFoodCode']:
        if col not in df_url.columns:
            df_url[col] = np.nan

    # Initialize retrieval score columns if not present
    if vectordb is not None:
        for col in ['retrieval_score_avg', 'retrieval_score_top1', 'retrieval_score_spread']:
            if col not in df_url.columns:
                df_url[col] = np.nan

    for num in range(last_num, num_iterations):
        start_index = last_index if num == last_num else 0
        for i in range(start_index, len(df_url)):
            url = df_url.loc[i, 'Link']
            food_code_gpt = df_url.loc[i, 'GPTFoodCode']
            if not pd.isna(food_code_gpt):
                food_code_gpt_first = food_code_gpt[2:10]
                if is_integer(food_code_gpt_first):
                    continue

            process_image_url(i, url, llm_vision, df_url, food_code_chain, results_path,
                              vectordb=vectordb, llm=llm, retrieval_prompt=retrieval_prompt)
            save_checkpoint(num, i + 1, checkpoint_path)  # Update checkpoint after each URL

        last_index = 0  # Reset last_index after completing each num iteration

def process_single_image_parallel(args):
    """Process a single image URL for parallel execution.

    Creates thread-local LLM instances for thread safety. The vectordb is
    shared (read-only) across threads.

    Args:
        args: Tuple of (index, url, vectordb)

    Returns:
        Tuple of (index, food_description, food_code, retrieval_scores, error)
    """
    index, url, vectordb, provider, chat_model, vision_model = args

    try:
        # Create thread-local LLM instances (NOT shared — thread safety)
        llm_vision = configure_chat_client(
            vision_model or MODELS['llm_vision'],
            provider,
            API_KEYS,
            temperature=0.1,
            max_tokens=1000,
        )
        llm = configure_chat_client(
            chat_model or MODELS['llm'],
            provider,
            API_KEYS,
            temperature=0.3,
        )

        # Create thread-local retriever and chain (vectordb is shared, read-only)
        retrieval_prompt = setup_retrieval_prompt()
        retriever = configure_retrievers(llm, vectordb, retrieval_prompt)
        food_code_chain = setup_code_prompt_chain(retriever, llm)

        # Validate image accessibility: local file or remote URL
        if os.path.isfile(url):
            image_uri = load_image_as_data_uri(url)
        else:
            req_url = requests.head(url, timeout=5)
            if req_url.status_code != 200:
                raise Exception("Image URL is not accessible")
            image_uri = url

        IMAGE_PROMPT = get_messages_from_url(image_uri)
        attempts = 0
        max_attempts = 5

        while True:
            try:
                image_response = llm_vision.invoke(IMAGE_PROMPT)
                food_description = image_response.content
                food_description_splits = food_description.split('\n')
                if food_description_splits[0] == "I can't help to analyze this image.":
                    raise Exception(food_description_splits[-1])
                break
            except Exception as e:
                if '429' in str(e) or 'rate limit' in str(e).lower():
                    if attempts < max_attempts - 1:
                        sleep_time = min(2 ** attempts * 30, 300)
                        logging.warning(f"Rate limit exceeded. Retrying in {sleep_time} seconds.")
                        attempts += 1
                        time.sleep(sleep_time)
                        continue
                    else:
                        logging.error("Maximum retry attempts reached.")
                        raise
                else:
                    logging.error(f"Failed to invoke llm_vision at index {index}: {e}")
                    raise

        food_code_response = food_code_chain.invoke({"question": food_description, "url_str": url})
        if food_code_response == "No appropriate food codes found from the context information.":
            raise Exception(food_code_response)

        # Compute retrieval scores
        scores = get_retrieval_scores_for_description(
            food_description, vectordb, llm, retrieval_prompt
        )

        return (index, food_description, str(food_code_response.split('\n')), scores, None)

    except Exception as e:
        return (index, str(e), None, None, str(e))


def process_image_urls_parallel(results_path, checkpoint_path, vectordb,
                                max_workers=5, num_iterations=5,
                                provider='openai', chat_model=None,
                                vision_model=None):
    """Process image URLs with parallel execution using ThreadPoolExecutor.

    Follows the same retry-iteration pattern as sequential processing but
    submits tasks to a thread pool for concurrent API calls.

    Args:
        results_path: Path to the results CSV file
        checkpoint_path: Path to the checkpoint file
        vectordb: Shared ChromaDB vector store (read-only)
        max_workers: Number of concurrent worker threads (default: 5)
        num_iterations: Number of retry iterations (default: 5)
    """
    df_url = pd.read_csv(results_path)
    last_num, last_index = load_checkpoint(checkpoint_path)

    # Initialize output columns if not present (first run)
    for col in ['GPTFoodDescription', 'GPTFoodCode']:
        if col not in df_url.columns:
            df_url[col] = np.nan

    # Initialize retrieval score columns if not present
    for col in ['retrieval_score_avg', 'retrieval_score_top1', 'retrieval_score_spread']:
        if col not in df_url.columns:
            df_url[col] = np.nan

    # Handle old checkpoint format for parallel mode
    start_iteration = last_num
    if last_index > 0 and last_index >= len(df_url):
        start_iteration = last_num + 1
        logging.info(f"Checkpoint indicates iteration {last_num} was completed.")

    logging.info(f"Starting parallel processing from iteration {start_iteration}")
    logging.info(f"Max workers: {max_workers}, Total iterations: {num_iterations}")

    for iteration in range(start_iteration, num_iterations):
        logging.info(f"ITERATION {iteration + 1}/{num_iterations}")

        # Reload dataframe for latest results when resuming
        if iteration > start_iteration or last_index > 0:
            df_url = pd.read_csv(results_path)

        # Collect tasks: images without valid food codes
        tasks = []
        for i in range(len(df_url)):
            url = df_url.loc[i, 'Link']
            food_code_gpt = df_url.loc[i, 'GPTFoodCode']
            if not pd.isna(food_code_gpt):
                food_code_gpt_first = str(food_code_gpt)[2:10]
                if is_integer(food_code_gpt_first):
                    continue
            tasks.append((i, url, vectordb, provider, chat_model, vision_model))

        logging.info(f"Found {len(tasks)} images to process in iteration {iteration + 1}")

        if not tasks:
            logging.info(f"No images to process in iteration {iteration + 1}. All done!")
            save_checkpoint(iteration + 1, 0, checkpoint_path)
            continue

        # Process in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(process_single_image_parallel, task): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                index, food_description, food_code, scores, error = future.result()
                completed += 1

                if error:
                    logging.warning(f"[Iter {iteration+1}][{completed}/{len(tasks)}] "
                                    f"Error at index {index}: {error}")
                    df_url.loc[index, 'GPTFoodDescription'] = food_description
                    df_url.loc[index, 'GPTFoodCode'] = np.nan
                else:
                    logging.info(f"[Iter {iteration+1}][{completed}/{len(tasks)}] "
                                 f"Completed index {index}")
                    df_url.loc[index, 'GPTFoodDescription'] = food_description
                    df_url.loc[index, 'GPTFoodCode'] = food_code
                    if scores:
                        for key, value in scores.items():
                            df_url.loc[index, key] = value

                # Save progress every 10 completed images
                if completed % 10 == 0:
                    df_url.to_csv(results_path, index=False)
                    save_checkpoint(iteration, completed, checkpoint_path)
                    logging.info(f"Progress saved: {completed}/{len(tasks)} "
                                 f"completed in iteration {iteration + 1}")

        # Save after each iteration
        df_url.to_csv(results_path, index=False)
        save_checkpoint(iteration + 1, 0, checkpoint_path)
        logging.info(f"Iteration {iteration + 1} complete! Processed {len(tasks)} images.")

    logging.info(f"All {num_iterations} iterations complete!")


def main(args):
    setup_logging(args.log_path)
    logging.info("Starting image processing script")
    llm, llm_vision, embedding = initialize_clients(
        MODELS,
        API_KEYS,
        provider=args.provider,
        chat_model=args.chat_model,
        vision_model=args.vision_model,
    )
    data = load_data(args.csv_file)
    vectordb = setup_vector_database(data, embedding)  # Ensure 'embedding' is initialized

    if args.parallel:
        logging.info(f"Using PARALLEL processing mode with {args.workers} workers")
        process_image_urls_parallel(
            args.results_file, args.checkpoint_file, vectordb,
            max_workers=args.workers,
            provider=args.provider,
            chat_model=args.chat_model,
            vision_model=args.vision_model,
        )
    else:
        logging.info("Using SEQUENTIAL processing mode")
        retrieval_prompt = setup_retrieval_prompt()
        retriever_from_llm = configure_retrievers(llm, vectordb, retrieval_prompt)
        food_code_chain = setup_code_prompt_chain(retriever_from_llm, llm)
        process_image_urls(args.results_file, args.checkpoint_file, llm_vision, food_code_chain,
                           vectordb=vectordb, llm=llm, retrieval_prompt=retrieval_prompt)
    
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    parser = argparse.ArgumentParser(description="Process image URLs and update with food descriptions.")
    parser.add_argument("--csv_file", required=True, help="Path to the CSV file containing data.")
    parser.add_argument("--checkpoint_file", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--results_file", required=True, help="Path to the results file containing image URLs.")
    parser.add_argument("--log_path", default="process_images.log", help="Path to the log file.")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing with ThreadPoolExecutor")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai",
                        help="Chat provider for food-description / code generation (default: openai)")
    parser.add_argument("--chat_model", default=None,
                        help="Override text chat model name")
    parser.add_argument("--vision_model", default=None,
                        help="Override multimodal vision model name")

    args = parser.parse_args()
    main(args)
