import asyncio
import os

from tinkoff.invest import AsyncClient
from tinkoff.invest.schemas import AssetsRequest, InstrumentStatus, InstrumentType
from dotenv import load_dotenv



load_dotenv()
TOKEN = os.getenv('TINKOFF_API_TOKEN')

async def main():
    async with AsyncClient(TOKEN) as client:
        r = await client.instruments.get_assets(
            request=AssetsRequest(
                instrument_type=InstrumentType.INSTRUMENT_TYPE_CURRENCY,
                instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE,
                 )  # pylint:disable=line-too-long
        )
        print("BASE SHARE ASSETS")
        for a in r.assets:
                for i in a.instruments:
                  print (a.name + " " + i.figi + " " + i.ticker)


if __name__ == "__main__":
    asyncio.run(main())
