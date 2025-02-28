import asyncio

from xzutils.AI.SentenceSimilarity import SentenceSimilarity

sentenceSimilarity = SentenceSimilarity()
sentenceSimilarity.init()


async def main():
    result = await sentenceSimilarity.get_similarity("你好", "你好吗")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
