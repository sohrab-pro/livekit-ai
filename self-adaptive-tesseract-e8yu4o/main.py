import logging
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero

# uncomment to enable Krisp BVC noise cancellation, currently supported on Linux and MacOS
# from livekit.plugins import noise_cancellation

## The storyteller agent is a multi-agent that can handoff the session to another agent.
## This example demonstrates more complex workflows with multiple agents.
## Each agent could have its own instructions, as well as different STT, LLM, TTS,
## or realtime models.

logger = logging.getLogger("multi-agent")

load_dotenv(dotenv_path=".env.local")

common_instructions = (
    "You are a helpful virtual assistant for a technology company. Your primary goal "
    "is to determine if the user wants to speak with a sales agent. You should ask "
    "if they want to be transferred to our sales team. Be polite and professional at all times."
)


@dataclass
class CharacterData:
    # Shared data that's used by the main agent.
    # This structure is passed as a parameter to function calls.

    name: Optional[str] = None
    query: Optional[str] = None


@dataclass
class StoryData:
    # Shared data that's used by the agents.
    # This structure is passed as a parameter to function calls.

    user_info: list[CharacterData] = field(default_factory=list)
    wants_transfer: bool = False
    query_type: Optional[str] = None


class LeadEditorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"{common_instructions} You are the initial contact point. "
            "Your job is to greet the user, ask if they want to be transferred to our sales agent, "
            "Do not interrupt the your response, Until you are done asking your question. "
            "and then transfer them if they say yes. If they say no, you MUST call the user_declines_transfer function. "
            "Be very attentive to any negative responses like 'no', 'nope', 'not interested', etc. - these all "
            "indicate the user is declining the transfer and you should call user_declines_transfer immediately. "
            "Immediately call user_declines_transfer if the user says no, not interested, or declines in any way. You should not wait for any other responses. "
            "If the user says yes, call user_wants_transfer. "
            "Start the conversation with a friendly greeting and immediately ask if they want "
            "to speak with a sales agent. Use a warm, approachable tone."
            "Never jump into other conversations than the one specifically about transferring the user to a sales agent. ",
            allow_interruptions=False
        )


    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()

    @function_tool
    async def user_introduction(
        self,
        context: RunContext[StoryData],
        name: str,
        query: str,
    ):
        """Called when the user has provided their information.

        Args:
            name: The name of the user
            query: What the user is interested in or asking about
        """

        user = CharacterData(name=name, query=query)
        context.userdata.user_info.append(user)

        logger.info(
            "added user info: %s with query: %s", name, query
        )

    @function_tool
    async def user_wants_transfer(
        self,
        context: RunContext[StoryData],
    ):
        """Called when the user has indicated they want to be transferred to a sales agent.
        """
        context.userdata.wants_transfer = True
        
        sales_agent = SpecialistEditorAgent("sales", chat_ctx=context.session._chat_ctx)

        logger.info("transferring user to sales agent")
        return sales_agent, "Great! I'll transfer you to our sales agent now."

    @function_tool
    async def user_declines_transfer(
        self,
        context: RunContext[StoryData],
    ):
        """Called when the user has declined to be transferred to a sales agent.
        Call this function immediately when the user says no, not interested, or declines in any way.
        """
        
        logger.info("user declined transfer to sales agent, ending call")
        
        # interrupt any existing generation
        self.session.interrupt()
        
        # generate a goodbye message and hang up
        await self.session.generate_reply(
            instructions="thank the user for their time and end the conversation with a short, polite message", 
            allow_interruptions=False
        )
        
        try:
            # Get the job context
            job_ctx = get_job_context()
            logger.info(f"Attempting to close room: {job_ctx.room.name}")
            
            # Delete the room
            response = await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
            logger.info(f"Room deletion response: {response}")
        except Exception as e:
            logger.error(f"Failed to close room: {str(e)}")
            # Log failure but don't attempt other methods that don't exist


class SpecialistEditorAgent(Agent):
    def __init__(self, specialty: str, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions=f"{common_instructions} You are a {specialty} agent. "
            "You are knowledgeable about our products and services. Your goal is to help "
            "the user with their sales-related inquiries. Be professional, helpful, and "
            "try to address their needs efficiently. If they have questions about "
            "products, pricing, or purchasing, provide helpful information.",
            # each agent could override any of the model services, including mixing
            # realtime and non-realtime models
            tts=openai.TTS(voice="echo"),
            chat_ctx=chat_ctx,
            allow_interruptions=False
        )

    async def on_enter(self):
        # when the agent is added to the session, we'll initiate the conversation by
        # using the LLM to generate a reply
        self.session.generate_reply()

    @function_tool
    async def gather_user_info(
        self,
        context: RunContext[StoryData],
        name: str,
        query: str,
    ):
        """Called when the sales agent needs to gather more information from the user.

        Args:
            name: The name of the user
            query: The user's request or question
        """

        user = CharacterData(name=name, query=query)
        context.userdata.user_info.append(user)

        logger.info(
            "sales agent gathered info from user: %s with query: %s", name, query
        )

    @function_tool
    async def set_query_type(
        self,
        context: RunContext[StoryData],
        query_type: str,
    ):
        """Called to categorize the type of query.

        Args:
            query_type: The category of the user's query (product, pricing, support, etc.)
        """

        context.userdata.query_type = query_type

        logger.info(
            "sales agent categorized query as: %s", query_type
        )

    @function_tool
    async def conversation_finished(self, context: RunContext[StoryData]):
        """When the sales agent has finished helping the user, 
        they can end the conversation.
        """
        # interrupt any existing generation
        self.session.interrupt()

        # generate a goodbye message and hang up
        # awaiting it will ensure the message is played out before returning
        await self.session.generate_reply(
            instructions="thank the user for their time and end the conversation with a polite message", 
            allow_interruptions=False
        )

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession[StoryData](
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=openai.STT(),
        tts=openai.TTS(),
        userdata=StoryData(),
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=LeadEditorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
