import asyncio
import aiohttp
import argparse
from codetiming import Timer


async def get_request(i, q, session):
    while not q.empty():
        url, ref = await q.get()
        match i:
            case 1:
                url += f":8001#{ref}"
            case 2:
                url += f":8002#{ref}"

        async with session.get(url) as response:
            await response.json()
            print(url, response.status)


async def main(loops, num_workers):
    q = asyncio.Queue()
    for i in range(loops):
        await q.put(("http://localhost", i))

    async with aiohttp.ClientSession() as session:
        with Timer(text="\nTotal elapsed time: {:.1f}"):
            tasks = [asyncio.create_task(get_request(1, q, session))]
            if num_workers > 1:
                tasks.append(asyncio.create_task(get_request(2, q, session)))
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Balancing Client Script")

    parser.add_argument(
        "iter",
        help="Number of iterations",
        default=10,
        nargs="?",
    )

    parser.add_argument(
        "workers",
        help="Number of remote workers",
        default=1,
        nargs="?",
    )

    args = parser.parse_args()

    asyncio.run(main(int(args.iter), int(args.workers)))
