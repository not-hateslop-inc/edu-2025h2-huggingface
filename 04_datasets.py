from datasets import load_dataset

ds = load_dataset("nsmc", split="train[:1000]", trust_remote_code=True)
print(ds)
print(ds[0])
print(ds.to_pandas().label.value_counts())
