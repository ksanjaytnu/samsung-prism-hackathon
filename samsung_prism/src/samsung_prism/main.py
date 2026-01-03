from samsung_prism.crew import SamsungCompetitorIntelligenceCrew

import os



def run():
    print("\n=== SAMSUNG COMPETITOR INTELLIGENCE ===\n")
    crew = SamsungCompetitorIntelligenceCrew()
    result = crew.crew().kickoff()
    print("\n=== FINAL INTELLIGENCE OUTPUT ===\n")
    print(result)

if __name__ == "__main__":
    run()

