from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from transformers import AutoTokenizer
import secrets


def datatrove_tokenization_executor(
    hf_dataset_id,
    name,
    id_column,
    text_column,
    output_folder,
    tokenizer_id,
    eos_token,
    num_workers,
    shuffle=True,
    job_id=None,
):
    if not job_id:
        job_id = secrets.token_hex(8)

    # use only existing columns from fredzzp/fine_code
    # schema: type, lang, text, file_name, file_ext, file_size_in_byte, program_lang
    columns = [text_column]
    if id_column is not None:
        columns.append(id_column)

    pipeline = [
        HuggingFaceDatasetReader(
            dataset=hf_dataset_id,
            dataset_options={
                "split": "train",
                "name": name,          # "default" is correct for this dataset
                "columns": columns,    # no "id" here anymore
            },
            # limit=40000,
            text_key=text_column,
            id_key=id_column,         # we’ll use file_name as id
            streaming=True,
        ),
        DocumentTokenizer(
            output_folder=output_folder,
            tokenizer_name_or_path=tokenizer_id,
            eos_token=eos_token,
            batch_size=1000,
            max_tokens_per_file=int(1e8),
            seed=1998,
        ),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"logs_{job_id}/",
        tasks=num_workers * 10,
        workers=num_workers,
    )

    return executor


def main():
    hf_checkpoint = "Qwen/Qwen2.5-Coder-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)

    executor = datatrove_tokenization_executor(
        job_id="fine_code_tokenization",
        hf_dataset_id="fredzzp/fine_code",
        name="default",
        id_column="file_name",
        text_column="text",
        output_folder="./fine_code",
        tokenizer_id=hf_checkpoint,
        eos_token=tokenizer.eos_token,
        shuffle=False,
        num_workers=4,
    )
    executor.run()


if __name__ == "__main__":
    main()
