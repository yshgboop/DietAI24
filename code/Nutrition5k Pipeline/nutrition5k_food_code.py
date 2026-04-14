"""
Food code inference for Nutrition5k dataset.
Adapts the RAG food code pipeline to work with local Nutrition5k images.
"""
import pandas as pd
import numpy as np
import os
import time
import base64
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config import MODELS, API_KEYS


def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def configure_openai_chat(model_key, MODELS, API_KEYS):
    """Create a ChatOpenAI configuration based on the specified model key."""
    return ChatOpenAI(
        model=MODELS[model_key],
        openai_api_key=API_KEYS['openai'],
        temperature=0.1 if model_key == 'llm_vision' else 0.3,
        max_tokens=1000 if model_key == 'llm_vision' else None
    )


def initialize_clients(MODELS, API_KEYS):
    llm = configure_openai_chat('llm', MODELS, API_KEYS)
    llm_vision = configure_openai_chat('llm_vision', MODELS, API_KEYS)
    embedding = OpenAIEmbeddings(
        model=MODELS['embedding'],
        openai_api_key=API_KEYS['openai']
    )
    return llm, llm_vision, embedding


def load_data(file_path):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    return data


def setup_vector_database(data, embedding):
    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
        try:
            vectordb = Chroma.from_documents(documents=data, embedding=embedding, collection_name="openai_embed")
            return vectordb
        except Exception as e:
            logging.error(f"Failed to setup vector database: {e}")
            attempts += 1
            time.sleep(180)
    logging.critical("Failed to initialize vector database after multiple attempts.")
    raise Exception("Failed to initialize vector database after multiple attempts.")


def image_to_base64_url(image_path):
    """Convert a local image file to a base64 data URL."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Detect image type
    if image_path.lower().endswith('.png'):
        mime_type = 'image/png'
    elif image_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = 'image/jpeg'
    else:
        mime_type = 'image/png'  # default

    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')

    return f"data:{mime_type};base64,{encoded}"


def get_messages_from_image(image_base64_url):
    """Generate a sequence of messages for an image - modified to identify individual ingredients."""
    return [
        SystemMessage(
            content="You are an expert at analyzing images with computer vision. "
                    "I will present you with a picture of food, which might be placed on a plate, inside a spoon, "
                    "or contained within different vessels. Your job is to accurately identify each individual food ingredient visible in the image."
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": """Please identify ALL individual food ingredients visible in this image.

List each ingredient separately, one per line, in the following format:
ingredient_name_1
ingredient_name_2
ingredient_name_3
...

Important:
- List only what you can clearly see in the image
- Be specific about the ingredient (e.g., "chicken breast" not just "chicken", "white rice" not just "rice")
- Include preparation method if visible (e.g., "grilled chicken", "steamed broccoli")
- Do not list combined dishes (e.g., avoid "chicken salad", instead list "chicken", "lettuce", "tomato")
- If you can't identify ingredients in this image, simply respond with 'I can't help to analyze this image.' and provide the reason on a new line."""},
                {"type": "image_url", "image_url": {"url": image_base64_url}}
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


def load_checkpoint(checkpoint_path):
    """Load the last processed indices from a checkpoint file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as file:
            content = file.read().strip()
            if content:
                num, index = content.split(',')
                return int(num), int(index)
    return 0, 0


def save_checkpoint(num, index, checkpoint_path):
    """Save the current indices to a checkpoint file."""
    with open(checkpoint_path, 'w') as file:
        file.write(f"{num},{index}")
    logging.info(f"Checkpoint saved at num {num}, index {index}")


def process_single_image(index, image_path, llm_vision, df, food_code_chain, results_path):
    """Process a single local image and update the DataFrame with food descriptions and codes.
    Modified to handle individual ingredients."""
    logging.info(f"Processing image at index {index}: {image_path}")
    try:
        # Convert local image to base64
        image_base64_url = image_to_base64_url(image_path)

        IMAGE_PROMPT = get_messages_from_image(image_base64_url)
        attempts = 0
        max_attempts = 5

        # Get ingredient list from vision model
        while True:
            try:
                image_response = llm_vision.invoke(IMAGE_PROMPT)
                ingredients_response = image_response.content
                ingredients_splits = ingredients_response.split('\n')
                if ingredients_splits[0] == "I can't help to analyze this image.":
                    raise Exception(ingredients_splits[-1])
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

        # Parse ingredients (one per line)
        ingredients_list = [ing.strip() for ing in ingredients_splits if ing.strip()]

        if not ingredients_list:
            raise Exception(f"No ingredients identified. GPT response was: {ingredients_response[:100]}")

        # For each ingredient, find FNDDS code using RAG
        ingredient_codes = []
        for ingredient in ingredients_list:
            try:
                code_response = food_code_chain.invoke(ingredient)
                if code_response == "No appropriate food codes found from the context information.":
                    # Keep ingredient without code
                    ingredient_codes.append(f"{ingredient}: unknown")
                else:
                    # Extract first 8-digit code from response
                    codes = code_response.split('\n')
                    first_code = codes[0].strip() if codes else "unknown"
                    ingredient_codes.append(f"{ingredient}: {first_code}")
            except Exception as e:
                logging.warning(f"Failed to get code for ingredient '{ingredient}': {e}")
                ingredient_codes.append(f"{ingredient}: unknown")

        # Format output
        # GPTFoodDescription: Original multi-line ingredient list
        # GPTFoodCode: "ingredient1: code1; ingredient2: code2; ..."
        df.loc[index, 'GPTFoodDescription'] = ingredients_response
        df.loc[index, 'GPTFoodCode'] = "; ".join(ingredient_codes)

    except Exception as e:
        df.loc[index, 'GPTFoodDescription'] = str(e)
        df.loc[index, 'GPTFoodCode'] = np.nan
    finally:
        df.to_csv(results_path, index=False)
        logging.info(f"Data saved to {results_path}")


def process_single_image_parallel(args):
    """Process a single image - used for parallel processing.
    Modified to handle individual ingredients.

    Creates its own LLM instances to be thread-safe, but shares the vectordb.
    """
    index, image_path, dish_id, vectordb = args

    try:
        # Create thread-local LLM instances
        llm_vision = configure_openai_chat('llm_vision', MODELS, API_KEYS)
        llm = configure_openai_chat('llm', MODELS, API_KEYS)

        # Setup thread-local retriever and chain (vectordb is shared, read-only)
        retrieval_prompt = setup_retrieval_prompt()
        retriever_from_llm = configure_retrievers(llm, vectordb, retrieval_prompt)
        food_code_chain = setup_code_prompt_chain(retriever_from_llm, llm)

        # Convert local image to base64
        image_base64_url = image_to_base64_url(image_path)
        IMAGE_PROMPT = get_messages_from_image(image_base64_url)

        # Get ingredient list from vision model
        attempts = 0
        max_attempts = 5
        while True:
            try:
                image_response = llm_vision.invoke(IMAGE_PROMPT)
                ingredients_response = image_response.content
                ingredients_splits = ingredients_response.split('\n')
                if ingredients_splits[0] == "I can't help to analyze this image.":
                    raise Exception(ingredients_splits[-1])
                break
            except Exception as e:
                if '429' in str(e) or 'rate limit' in str(e).lower():
                    if attempts < max_attempts - 1:
                        sleep_time = min(2 ** attempts * 30, 300)
                        attempts += 1
                        time.sleep(sleep_time)
                        continue
                    else:
                        raise
                else:
                    raise

        # Parse ingredients (one per line)
        ingredients_list = [ing.strip() for ing in ingredients_splits if ing.strip()]

        if not ingredients_list:
            raise Exception(f"No ingredients identified. GPT response was: {ingredients_response[:100]}")

        # For each ingredient, find FNDDS code using RAG
        ingredient_codes = []
        for ingredient in ingredients_list:
            try:
                code_response = food_code_chain.invoke(ingredient)
                if code_response == "No appropriate food codes found from the context information.":
                    ingredient_codes.append(f"{ingredient}: unknown")
                else:
                    # Extract first 8-digit code from response
                    codes = code_response.split('\n')
                    first_code = codes[0].strip() if codes else "unknown"
                    ingredient_codes.append(f"{ingredient}: {first_code}")
            except Exception as e:
                ingredient_codes.append(f"{ingredient}: unknown")

        # Format output
        food_description = ingredients_response
        food_code = "; ".join(ingredient_codes)

        return index, dish_id, food_description, food_code, None

    except Exception as e:
        return index, dish_id, str(e), None, str(e)


def process_nutrition5k_images_parallel(df, image_base_path, results_path, checkpoint_path, vectordb, max_workers=5, num_iterations=5):
    """Process Nutrition5k images with parallel processing.

    Runs multiple iterations to retry failed images (same as sequential version).

    Args:
        df: DataFrame with dish data
        image_base_path: Base path to images
        results_path: Path to save results
        checkpoint_path: Path to checkpoint file
        vectordb: Shared vector database (read-only)
        max_workers: Number of concurrent threads (default 5)
        num_iterations: Number of retry iterations (default 5)
    """
    last_num, last_index = load_checkpoint(checkpoint_path)

    # Handle old checkpoint format: if last_index > 0, it means the old code
    # completed that many items in iteration last_num. For parallel mode,
    # we treat this as "iteration last_num completed" if last_index >= total items
    # or "iteration last_num in progress" otherwise.
    # Since parallel processes all items at once, if we have any last_index > 0,
    # we should check if iteration was effectively completed by checking tasks.
    start_iteration = last_num
    if last_index > 0 and last_index >= len(df):
        # Old format: iteration completed, move to next
        start_iteration = last_num + 1
        print(f"Detected old checkpoint format. Iteration {last_num} was completed.")

    print(f"Starting parallel processing from iteration {start_iteration}")
    print(f"Max workers: {max_workers}, Total iterations: {num_iterations}")

    for iteration in range(start_iteration, num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # Reload dataframe to get latest results (in case of resume)
        # Always reload if resuming (last_index > 0) or if not first iteration
        if iteration > start_iteration or last_index > 0:
            df = pd.read_csv(results_path)
            print(f"Reloaded dataframe with latest results")

        # Prepare tasks for images that need processing
        tasks = []
        for i in range(len(df)):
            # Skip if already has valid food code (new format: "ingredient: code; ...")
            food_code_gpt = df.loc[i, 'GPTFoodCode']
            if not pd.isna(food_code_gpt):
                food_code_str = str(food_code_gpt)
                # Check for new format: contains ":" (ingredient: code)
                # Skip rows where ingredients have been identified and coded
                if ':' in food_code_str and len(food_code_str) > 5:
                    continue

            dish_id = df.loc[i, 'dish_id']
            image_path = os.path.join(image_base_path, dish_id, 'rgb.png')

            if not os.path.exists(image_path):
                logging.warning(f"Image not found for dish_id {dish_id}: {image_path}")
                df.loc[i, 'GPTFoodDescription'] = "Image not found"
                df.loc[i, 'GPTFoodCode'] = np.nan
                continue

            tasks.append((i, image_path, dish_id, vectordb))

        print(f"Found {len(tasks)} images to process in iteration {iteration + 1}")

        if not tasks:
            print(f"No images to process in iteration {iteration + 1}. All done!")
            save_checkpoint(iteration + 1, 0, checkpoint_path)
            continue

        # Process in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_single_image_parallel, task): task for task in tasks}

            for future in as_completed(future_to_task):
                index, dish_id, food_description, food_code, error = future.result()
                completed += 1

                if error:
                    print(f"[Iter {iteration+1}][{completed}/{len(tasks)}] Error at index {index} ({dish_id}): {error}")
                    df.loc[index, 'GPTFoodDescription'] = food_description
                    df.loc[index, 'GPTFoodCode'] = np.nan
                else:
                    print(f"[Iter {iteration+1}][{completed}/{len(tasks)}] Completed index {index} ({dish_id})")
                    df.loc[index, 'GPTFoodDescription'] = food_description
                    df.loc[index, 'GPTFoodCode'] = food_code

                # Save progress periodically (every 10 images)
                if completed % 10 == 0:
                    df.to_csv(results_path, index=False)
                    # Save checkpoint with current iteration and completed count
                    # Format: iteration, completed_in_this_iteration
                    save_checkpoint(iteration, completed, checkpoint_path)
                    print(f"Progress saved: {completed}/{len(tasks)} completed in iteration {iteration + 1}")

        # Save after each iteration
        df.to_csv(results_path, index=False)
        save_checkpoint(iteration + 1, 0, checkpoint_path)
        print(f"Iteration {iteration + 1} complete! Processed {len(tasks)} images.")

    print(f"\n{'='*60}")
    print(f"All {num_iterations} iterations complete!")
    print(f"{'='*60}")


def process_nutrition5k_images(df, image_base_path, results_path, checkpoint_path, llm_vision, food_code_chain):
    """Process Nutrition5k images with checkpointing."""
    num_iterations = 5
    last_num, last_index = load_checkpoint(checkpoint_path)

    print(f"Starting from checkpoint: iteration {last_num}, index {last_index}")
    print(f"Total images to process: {len(df)}")

    for num in range(last_num, num_iterations):
        start_index = last_index if num == last_num else 0
        for i in range(start_index, len(df)):
            dish_id = df.loc[i, 'dish_id']
            food_code_gpt = df.loc[i, 'GPTFoodCode']

            # Skip if already has valid food code (new format: "ingredient: code; ...")
            if not pd.isna(food_code_gpt):
                food_code_str = str(food_code_gpt)
                # Check for new format: contains ":" (ingredient: code)
                if ':' in food_code_str and len(food_code_str) > 5:
                    continue

            # Construct image path
            image_path = os.path.join(image_base_path, dish_id, 'rgb.png')

            if not os.path.exists(image_path):
                logging.warning(f"Image not found for dish_id {dish_id}: {image_path}")
                df.loc[i, 'GPTFoodDescription'] = "Image not found"
                df.loc[i, 'GPTFoodCode'] = np.nan
                df.to_csv(results_path, index=False)
                save_checkpoint(num, i + 1, checkpoint_path)
                continue

            print(f"[Iteration {num+1}/5] Processing {i+1}/{len(df)}: {dish_id}")
            process_single_image(i, image_path, llm_vision, df, food_code_chain, results_path)
            save_checkpoint(num, i + 1, checkpoint_path)

        last_index = 0

    print("Processing complete!")


def main(args):
    setup_logging(args.log_path)
    logging.info("Starting Nutrition5k food code inference")

    # Initialize LLM clients
    llm, llm_vision, embedding = initialize_clients(MODELS, API_KEYS)

    # Load FNDDS data for RAG
    print(f"Loading FNDDS data from: {args.fndds_file}")
    data = load_data(args.fndds_file)

    # Setup vector database
    print("Setting up vector database...")
    vectordb = setup_vector_database(data, embedding)

    # Load Nutrition5k data
    print(f"Loading Nutrition5k data from: {args.nutrition5k_file}")
    df = pd.read_csv(args.nutrition5k_file)

    # Add columns for results if not present (use object dtype for string data)
    if 'GPTFoodDescription' not in df.columns:
        df['GPTFoodDescription'] = pd.Series([np.nan] * len(df), dtype='object')
    if 'GPTFoodCode' not in df.columns:
        df['GPTFoodCode'] = pd.Series([np.nan] * len(df), dtype='object')

    # Save initial results file
    df.to_csv(args.results_file, index=False)

    # Process images
    if args.parallel:
        print(f"\n=== Using PARALLEL processing mode with {args.max_workers} workers ===")
        print("This is much faster but may hit rate limits. Reduce --max_workers if needed.\n")
        process_nutrition5k_images_parallel(
            df=df,
            image_base_path=args.image_base_path,
            results_path=args.results_file,
            checkpoint_path=args.checkpoint_file,
            vectordb=vectordb,
            max_workers=args.max_workers
        )
    else:
        print("\n=== Using SEQUENTIAL processing mode ===")
        print("Slower but safer for rate limits.\n")
        # Setup retrieval chain for sequential mode
        retrieval_prompt = setup_retrieval_prompt()
        retriever_from_llm = configure_retrievers(llm, vectordb, retrieval_prompt)
        food_code_chain = setup_code_prompt_chain(retriever_from_llm, llm)

        process_nutrition5k_images(
            df=df,
            image_base_path=args.image_base_path,
            results_path=args.results_file,
            checkpoint_path=args.checkpoint_file,
            llm_vision=llm_vision,
            food_code_chain=food_code_chain
        )

    print("\n=== Processing complete! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food code inference for Nutrition5k dataset")
    parser.add_argument("--fndds_file", required=True,
                        help="Path to FNDDS CSV file for RAG")
    parser.add_argument("--nutrition5k_file", required=True,
                        help="Path to processed Nutrition5k metadata CSV")
    parser.add_argument("--image_base_path", required=True,
                        help="Base path to Nutrition5k images (e.g., /Volumes/My Passport/nutrition5k_dataset/imagery/realsense_overhead)")
    parser.add_argument("--results_file", required=True,
                        help="Path to save results CSV")
    parser.add_argument("--checkpoint_file", required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--log_path", default="nutrition5k_food_code.log",
                        help="Path to log file")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel processing (faster but may hit rate limits)")
    parser.add_argument("--max_workers", type=int, default=5,
                        help="Number of parallel workers (default: 5)")

    args = parser.parse_args()
    main(args)
