import os
import argparse


import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from prompts import specialist_prompt, reviewer_prompt


# Load environment variables
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Generate corrected sentences using Upstage API")
    parser.add_argument("--input", default="data/train_dataset.csv", help="Input CSV path containing err_sentence column")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--model", default="solar-pro2", help="Model name (default: solar-pro2)")
    args = parser.parse_args()


    # Load data
    df = pd.read_csv(args.input)
   
    if "err_sentence" not in df.columns:
        raise ValueError("Input CSV must contain 'err_sentence' column")
   
    if "id" not in df.columns:
        print("Warning: 'id' column not found. Generating without id.")
        df['id'] = [f'temp_id_{i}' for i in range(len(df))]
       
    # Setup Upstage client
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY not found in environment variables")
   
    client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
   
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")


    err_sentences = []
    cor_sentences = []
    ids = []
   
    # Process each sentence
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating (Safe 2-Call)"):
        text = str(row["err_sentence"])
       
        ids.append(row["id"])
        err_sentences.append(text)
       
        corrected_1 = ""
        corrected_2 = ""
       
        try:
            # --- API 호출 1: 전문가 (Specialist) ---
            prompt_1 = specialist_prompt.format(text=text)
            resp_1 = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "당신은 한국어 문장 교정 전문가입니다. 맞춤법/띄어쓰기/문장부호/문법을 자연스럽게 교정하세요. 반드시 불필한 설명 없이 교정된 문장만 출력하세요."},
                    {"role": "user", "content": prompt_1},
                ],
                temperature=0.0,
            )
            corrected_1 = resp_1.choices[0].message.content.strip()


            # --- API 호출 2: 검토자 (Reviewer) ---
            prompt_2 = reviewer_prompt.format(text=corrected_1)
            resp_2 = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "당신은 1차 교정본을 검토하는 2차 검토자입니다. 명백한 오류가 아니라면 절대 수정하지 마세요. 설명 없이 교정된 문장만 출력하세요."},
                    {"role": "user", "content": prompt_2},
                ],
                temperature=0.0,
            )
            corrected_2 = resp_2.choices[0].message.content.strip()
           
            cor_sentences.append(corrected_2)
           
        except Exception as e:
            print(f"Error processing: {text[:50]}... - {e}")
            cor_sentences.append(corrected_1 if corrected_1 else text) # 오류 시 1차 교정본 사용


    # Save results with required column names
    out_df = pd.DataFrame({
        "id": ids,
        "err_sentence": err_sentences,
        "cor_sentence": cor_sentences
    })
   
    out_df = out_df[["id", "err_sentence", "cor_sentence"]]
   
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")



if __name__ == "__main__":
    main()
