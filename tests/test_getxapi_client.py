import os
import unittest
from unittest.mock import Mock, patch

from src.api import getxapi_client


class GetXAPIClientTests(unittest.TestCase):
    def test_get_headers_uses_bearer_token(self):
        with patch.dict(os.environ, {"GETXAPI_KEY": " test-key "}, clear=True):
            self.assertEqual(
                getxapi_client.get_headers(),
                {
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json",
                },
            )

    def test_fetch_replies_calls_endpoint_with_cursor(self):
        response = Mock()
        response.json.return_value = {"tweetId": "123", "replies": []}
        response.raise_for_status.return_value = None

        with patch.dict(os.environ, {"GETXAPI_KEY": "test-key"}, clear=True):
            with patch.object(getxapi_client.requests, "get", return_value=response) as get:
                payload = getxapi_client.fetch_replies("123", cursor="abc")

        self.assertEqual(payload, {"tweetId": "123", "replies": []})
        get.assert_called_once_with(
            "https://api.getxapi.com/twitter/tweet/replies",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            params={"tweetId": "123", "cursor": "abc"},
            timeout=30,
        )

    def test_fetch_all_replies_follows_next_cursor(self):
        pages = [
            {"tweetId": "123", "replies": [{"id": "1"}], "has_more": True, "next_cursor": "next"},
            {"tweetId": "123", "replies": [{"id": "2"}], "has_more": False},
        ]

        with patch.object(getxapi_client, "fetch_replies", side_effect=pages) as fetch:
            replies = getxapi_client.fetch_all_replies("123")

        self.assertEqual(replies, [{"id": "1"}, {"id": "2"}])
        self.assertEqual(fetch.call_args_list[0].kwargs, {"tweet_id": "123", "cursor": None})
        self.assertEqual(fetch.call_args_list[1].kwargs, {"tweet_id": "123", "cursor": "next"})

    def test_normalize_replies_matches_project_schema(self):
        response = {
            "tweetId": "parent-1",
            "replies": [
                {
                    "id": "reply-1",
                    "createdAt": "2026-06-11T20:01:00.000Z",
                    "url": "https://x.com/example/status/reply-1",
                    "text": "What a finish!",
                    "author": {"userName": "fan", "name": "Football Fan"},
                    "likeCount": 3,
                    "replyCount": 1,
                    "viewCount": 100,
                }
            ],
        }

        rows = getxapi_client.normalize_replies(
            response,
            match="Canada vs France",
            event="goal",
            source_account="FIFAWorldCup",
            team="Canada",
            player="Davies",
        )

        self.assertEqual(
            rows,
            [
                {
                    "tweet_id": "reply-1",
                    "parent_tweet_id": "parent-1",
                    "url": "https://x.com/example/status/reply-1",
                    "timestamp": "2026-06-11T20:01:00.000Z",
                    "text": "What a finish!",
                    "author_username": "fan",
                    "author_name": "Football Fan",
                    "like_count": 3,
                    "reply_count": 1,
                    "view_count": 100,
                    "match": "Canada vs France",
                    "team": "Canada",
                    "player": "Davies",
                    "event": "goal",
                    "source_account": "FIFAWorldCup",
                    "source": "getxapi",
                }
            ],
        )

    def test_normalize_replies_accepts_nested_data_shape(self):
        rows = getxapi_client.normalize_replies(
            {"data": {"tweet_id": "parent-1", "replies": [{"tweet_id": "reply-1"}]}},
            match="Canada vs France",
            event="goal",
        )

        self.assertEqual(rows[0]["tweet_id"], "reply-1")
        self.assertEqual(rows[0]["parent_tweet_id"], "parent-1")


if __name__ == "__main__":
    unittest.main()
