select game_id, team_id, participant_id, champion_name, win
from participants
where
    summoner_name in(
        select summoner_name
        from league_item
        where
            league_id = '04a0f903-37c7-316f-af90-5f8eed42011c'
    );