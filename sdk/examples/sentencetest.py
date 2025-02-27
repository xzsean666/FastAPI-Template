from sdk import SentenceSimilarity


async def main():
    sentence_similarity = SentenceSimilarity(
        model_path="/home/sean/AI/universal-sentence-encoder-v2"
    )
    sentence_similarity.init()
    result = await sentence_similarity.get_similarity(
        "Hello, world!", "Hello, world again!"
    )
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
