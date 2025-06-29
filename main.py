from AI_interview import team_Config, interview
import asyncio

async def main():
    team = await team_Config()

    async for message in interview(team):
        print('-'*70)
        print(message)

if __name__=="__main__":
    asyncio.run(main())