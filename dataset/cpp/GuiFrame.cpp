#include "PrecompiledHeader.h"

#include "GameApp.h"
#include "GuiFrame.h"

GuiFrame::GuiFrame(GameObserver* observer)
    : GuiLayer(observer)
{
    //if (observer->getResourceManager())
    //{
        /*if (observer->getResourceManager()->GetTexture("wood.png"))
        {
            wood = observer->getResourceManager()->RetrieveQuad("wood.png", 0, 0, 0, 0);
            wood->mHeight = 32.f;
            wood->mWidth = 480.f;
        }
        else
        {
            GameApp::systemError += "Can't load wood texture : " __FILE__ "\n";
        }*/

        /*if (observer->getResourceManager()->GetTexture("gold.png"))
        {
            gold1 = observer->getResourceManager()->RetrieveQuad("gold.png", 0, 0, SCREEN_WIDTH, 6, "gold1");
            gold2 = observer->getResourceManager()->RetrieveQuad("gold.png", 0, 6, SCREEN_WIDTH, 6, "gold2");
            if (observer->getResourceManager()->GetTexture("goldglow.png"))
                goldGlow = observer->getResourceManager()->RetrieveQuad("goldglow.png", 1, 1, SCREEN_WIDTH - 2, 18);
            if (gold2)
            {
                gold2->SetColor(ARGB(127, 255, 255, 255));
                gold2->SetHFlip(true);
            }
        }*/
    //}
    //step = 0.0;

}

GuiFrame::~GuiFrame()
{
}

void GuiFrame::Render()
{
    /*JRenderer* renderer = JRenderer::GetInstance();
    float sized = step / 4;
    if (sized > SCREEN_WIDTH)
        sized -= SCREEN_WIDTH;
    renderer->RenderQuad(wood.get(), 0, 0);*/
    /*if (gold1.get())
    {
        renderer->RenderQuad(gold1.get(), -sized, 16);
        renderer->RenderQuad(gold1.get(), -sized + 479, 16);

        if (goldGlow.get())
        {
            goldGlow->SetColor(ARGB((100+(rand()%50)), 255, 255, 255));
            renderer->RenderQuad(goldGlow.get(), -sized, 9);
            renderer->RenderQuad(goldGlow.get(), -sized + 480, 9);
        }

        if (gold2.get())
        {
            renderer->RenderQuad(gold2.get(), step / 2, 16);
            renderer->RenderQuad(gold2.get(), step / 2 - 479, 16);
        }
    }*/
}

void GuiFrame::Update(float dt)
{
    /*step += dt * 5;
    if (step > 2 * SCREEN_WIDTH)
        step -= 2 * SCREEN_WIDTH;*/
}
