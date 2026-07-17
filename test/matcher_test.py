import asyncio

from agents.Matcher import PaperMatcher


async def main():
    paper_matcher = PaperMatcher()
    result = await paper_matcher.match("GTLR-GS")
    result = await paper_matcher.download(result)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())